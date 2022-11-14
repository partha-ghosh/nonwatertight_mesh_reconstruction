from argparse import _VersionAction
import time, os
import numpy as np
import torch
from torch.nn import functional as F
import trimesh

from src.dpsr import DPSR
from src.model import PSR2Mesh
from src.utils import grid_interp, verts_on_largest_mesh,\
                    export_pointcloud, mc_from_psr, GaussianSmoothing
from src.visualize import visualize_points_mesh, visualize_psr_grid, \
                    visualize_mesh_phong, render_rgb
from torchvision.utils import save_image
from torchvision.io import write_video
from pytorch3d.loss import chamfer_distance
import open3d as o3d

import cv2

import matplotlib.pyplot as plt

class Trainer(object):
    '''
    Args:
        cfg       : config file
        optimizer : pytorch optimizer object
        device    : pytorch device
    '''

    def __init__(self, cfg, optimizer, device=None):
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.psr2mesh = PSR2Mesh.apply
        self.data_type = cfg['data']['data_type']

        # initialize DPSR
        self.dpsr = DPSR(res=(cfg['model']['grid_res'], 
                            cfg['model']['grid_res'], 
                            cfg['model']['grid_res']), 
                        sig=cfg['model']['psr_sigma'])
        if torch.cuda.device_count() > 1:    
            self.dpsr = torch.nn.DataParallel(self.dpsr) # parallell DPSR
        self.dpsr = self.dpsr.to(device)
        self.pcv = None
        
    def train_step(self, data, inputs, model, it):
        ''' Performs a training step.

        Args:
            data (dict)              : data dictionary
            inputs (torch.tensor)    : input point clouds
            model (nn.Module or None): a neural network or None
            it (int)                 : the number of iterations
        '''

        if self.pcv is None:
            self.gen_envelope(data)

        self.optimizer.zero_grad()
        loss, loss_each = self.compute_loss(inputs, data, model, it)

        loss.backward()
        self.optimizer.step()
        
        return loss.item(), loss_each

    def compute_loss(self, inputs, data, model, it=0):
        '''  Compute the loss.
        Args:
            data (dict)              : data dictionary
            inputs (torch.tensor)    : input point clouds
            model (nn.Module or None): a neural network or None
            it (int)                 : the number of iterations
        '''

        device = self.device
        res = self.cfg['model']['grid_res']
        
        # source oriented point clouds to PSR grid
        psr_grid, points, normals = self.pcl2psr(inputs)
        # psr_grid = (psr_grid - torch.min(psr_grid))/(torch.max(psr_grid)-torch.min(psr_grid))
        # psr_grid = (1 - psr_grid)
        # psr_grid = 1 / (1 + torch.exp(-100*(psr_grid-0.5)))

        # build mesh
        PSR2Mesh.use_mask(torch.where(self.pcv==1, True, False).cpu().numpy())
        v, f, n = self.psr2mesh(psr_grid)
        
        # the output is in the range of [0, 1), we make it to the real range [0, 1]. 
        # This is a hack for our DPSR solver
        v = v * res / (res-1) 

        points = points * 2. - 1.
        v = v * 2. - 1. # within the range of (-1, 1)

        loss = 0
        loss_each = {}
        # compute loss
        if self.data_type == 'point':
            if self.cfg['train']['w_chamfer'] > 0:
                loss_ = self.cfg['train']['w_chamfer'] * \
                        self.compute_3d_loss(v, data)
                loss_each['chamfer'] = loss_
                loss += loss_
        elif self.data_type == 'img': 
            loss, loss_each = self.compute_2d_loss(inputs, data, model)

        return loss, loss_each


    def pcl2psr(self, inputs):
        '''  Convert an oriented point cloud to PSR indicator grid
        Args:
            inputs (torch.tensor): input oriented point clouds
        '''

        points, normals = inputs[...,:3], inputs[...,3:]
        if self.cfg['model']['apply_sigmoid']:
            points = torch.sigmoid(points)
        if self.cfg['model']['normal_normalize']:
            normals = normals / normals.norm(dim=-1, keepdim=True)

        # DPSR to get grid
        psr_grid = self.dpsr(points, normals).unsqueeze(1)
        psr_grid = torch.tanh(psr_grid)

        return psr_grid, points, normals

    def compute_3d_loss(self, v, data):
        '''  Compute the loss for point clouds.
        Args:
            v (torch.tensor)         : mesh vertices
            data (dict)              : data dictionary
        '''

        pts_gt = data.get('target_points')
        idx = np.random.randint(pts_gt.shape[1], size=self.cfg['train']['n_sup_point'])
        if self.cfg['train']['subsample_vertex']:
            #chamfer distance only on random sampled vertices
            idx = np.random.randint(v.shape[1], size=self.cfg['train']['n_sup_point'])
            loss, _ = chamfer_distance(v[:, idx], pts_gt)
        else:
            loss, _ = chamfer_distance(v, pts_gt)

        return loss
    
    def compute_2d_loss(self, inputs, data, model):
        '''  Compute the 2D losses.
        Args:
            inputs (torch.tensor)    : input source point clouds
            data (dict)              : data dictionary
            model (nn.Module or None): neural network or None
        '''
        
        losses = {"color": 
                    {"weight": self.cfg['train']['l_weight']['rgb'], 
                     "values": []
                    },
                  "silhouette": 
                    {"weight": self.cfg['train']['l_weight']['mask'], 
                     "values": []},
                }
        loss_all = {k: torch.tensor(0.0, device=self.device) for k in losses}
            
        # forward pass
        out = model(inputs, data)

        if out['rgb'] is not None:
            rgb_gt = out['rgb_gt'].reshape(self.cfg['data']['n_views_per_iter'], 
                                                -1, 3)[out['vis_mask']]
            loss_all["color"] += torch.nn.L1Loss(reduction='sum')(rgb_gt,
                                        out['rgb']) / out['rgb'].shape[0]

        if out['mask'] is not None:
            loss_all["silhouette"] += ((out['mask'] - out['mask_gt']) ** 2).mean()  
            
        # weighted sum of the losses
        loss = torch.tensor(0.0, device=self.device)
        for k, l in loss_all.items():
            loss += l * losses[k]["weight"]
            losses[k]["values"].append(l)

        return loss, loss_all

    def point_resampling(self, inputs):
        '''  Resample points
        Args:
            inputs (torch.tensor): oriented point clouds
        '''
    
        psr_grid, points, normals = self.pcl2psr(inputs)
        
        # shortcuts
        n_grow = self.cfg['train']['n_grow_points']

        # [hack] for points resampled from the mesh from marching cubes, 
        # we need to divide by s instead of (s-1), and the scale is correct.
        verts, faces, _ = mc_from_psr(psr_grid, real_scale=False, zero_level=0)

        # find the largest component
        pts_mesh, faces_mesh = verts_on_largest_mesh(verts, faces)
    
        # sample vertices only from the largest component, not from fragments
        mesh = trimesh.Trimesh(vertices=pts_mesh, faces=faces_mesh)
        pi, face_idx = mesh.sample(n_grow+points.shape[1], return_index=True)
        normals_i = mesh.face_normals[face_idx].astype('float32')
        pts_mesh = torch.tensor(pi.astype('float32')).to(self.device)[None]
        n_mesh = torch.tensor(normals_i).to(self.device)[None]

        points, normals = pts_mesh, n_mesh
        print('{} total points are resampled'.format(points.shape[1]))
    
        # update inputs
        points = torch.log(points / (1 - points)) # inverse sigmoid
        inputs = torch.cat([points, normals], dim=-1)
        inputs.requires_grad = True  

        return inputs

    def visualize(self, data, inputs, renderer, epoch, o3d_vis=None):
        '''  Visualization.
        Args:
            data (dict)                 : data dictionary
            inputs (torch.tensor)       : source point clouds
            renderer (nn.Module or None): a neural network or None
            epoch (int)                 : the number of iterations
            o3d_vis (o3d.Visualizer)    : open3d visualizer
        '''
        
        data_type = self.cfg['data']['data_type']
        it = '{:04d}'.format(int(epoch/self.cfg['train']['visualize_every']))

        
        if (self.cfg['train']['exp_mesh']) \
         | (self.cfg['train']['exp_pcl']) \
         | (self.cfg['train']['o3d_show']):
            psr_grid, points, normals = self.pcl2psr(inputs)

            with torch.no_grad():
                v, f, n = mc_from_psr(psr_grid, pytorchify=True,
                zero_level=self.cfg['data']['zero_level'], real_scale=True)
                v, f, n = v[None], f[None], n[None]
        
                v = v * 2. - 1. # change to the range of [-1, 1]

            color_v = None
            if data_type == 'img':
                if self.cfg['train']['vis_vert_color'] & \
                    (self.cfg['train']['l_weight']['rgb'] != 0.):
                    color_v = renderer['color'](v, n).squeeze().detach().cpu().numpy()
                    color_v[color_v<0], color_v[color_v>1] = 0., 1.

            vv = v.detach().squeeze().cpu().numpy()
            ff = f.detach().squeeze().cpu().numpy()
            points = points * 2 - 1
            visualize_points_mesh(o3d_vis, points, normals, 
                                vv, ff, self.cfg, it, epoch, color_v=color_v)

        else:
            v, f, n = inputs

        
        if (data_type == 'img') & (self.cfg['train']['vis_rendering']):
            pred_imgs = []
            pred_masks = []
            n_views = len(data['poses'])
            # idx_list = trange(n_views)
            idx_list = [13, 24, 27, 48]

            #! 
            model = renderer.eval()
            for idx in idx_list:
                pose = data['poses'][idx]
                rgb = data['rgbs'][idx]
                mask_gt = data['masks'][idx]
                img_size = rgb.shape[0] if rgb.shape[0]== rgb.shape[1] else (rgb.shape[0], rgb.shape[1])
                ray = None
                if 'rays' in data.keys():
                    ray = data['rays'][idx]
                if self.cfg['train']['l_weight']['rgb'] != 0.:
                    fea_grid = None
                    if model.unet3d is not None:
                        with torch.no_grad():
                            fea_grid = model.unet3d(psr_grid).permute(0, 2, 3, 4, 1)
                    if model.encoder is not None:
                        pp = torch.cat([(points+1)/2, normals], dim=-1)
                        fea_grid = model.encoder(pp, 
                                    normalize=False).permute(0, 2, 3, 4, 1)

                    pred, visible_mask = render_rgb(v, f, n, pose, 
                                            model.rendering_network.eval(), 
                                            img_size, ray=ray, fea_grid=fea_grid)
                    img_pred = torch.zeros([rgb.shape[0]*rgb.shape[1], 3])
                    img_pred[visible_mask] = pred.detach().cpu()

                    img_pred = img_pred.reshape(rgb.shape[0], rgb.shape[1], 3)
                    img_pred[img_pred<0], img_pred[img_pred>1] = 0., 1.
                    filename=os.path.join(self.cfg['train']['dir_rendering'], 
                                            'rendering_{}_{:d}.png'.format(it, idx))
                    save_image(img_pred.permute(2, 0, 1), filename)
                    pred_imgs.append(img_pred)

                #! Mesh rendering using Phong shading model
                filename=os.path.join(self.cfg['train']['dir_rendering'], 
                                            'mesh_{}_{:d}.png'.format(it, idx))
                visualize_mesh_phong(v, f, n, pose, img_size, name=filename)

            if len(pred_imgs) >= 1:
                pred_imgs = torch.stack(pred_imgs, dim=0)
                save_image(pred_imgs.permute(0, 3, 1, 2), 
                            os.path.join(self.cfg['train']['dir_rendering'], 
                                                '{}.png'.format(it)), nrow=4)
                if self.cfg['train']['save_video']:
                    write_video(os.path.join(self.cfg['train']['dir_rendering'], 
                                        '{}.mp4'.format(it)), 
                                        (pred_imgs*255.).type(torch.uint8), fps=24)

    def save_mesh_pointclouds(self, inputs, epoch, center=None, scale=None):
        '''  Save meshes and point clouds.
        Args:
            inputs (torch.tensor)       : source point clouds
            epoch (int)                 : the number of iterations
            center (numpy.array)        : center of the shape
            scale (numpy.array)         : scale of the shape
        '''

        exp_pcl = self.cfg['train']['exp_pcl']
        exp_mesh = self.cfg['train']['exp_mesh']
        
        psr_grid, points, normals = self.pcl2psr(inputs)
        
        #--------------------------------------------
        s = psr_grid.shape[-1] # size of psr_grid
        # for i in range(s):
        #     psr_grid[0][0][i] = torch.where(self.pcv[i]==1, psr_grid[0][0][i], torch.tensor([1], dtype=torch.float).to(self.device))
        for i in range(s):
            plt.imsave(f'out/x/{i}.png', psr_grid[0][0][i].clone().detach().cpu(), cmap=plt.cm.gray)
        input("Done >>>")

        if exp_pcl:
            dir_pcl = self.cfg['train']['dir_pcl']
            p = points.squeeze(0).detach().cpu().numpy()
            p = p * 2 - 1
            n = normals.squeeze(0).detach().cpu().numpy()
            if scale is not None:
                p *= scale
            if center is not None:
                p += center
            export_pointcloud(os.path.join(dir_pcl, '{:04d}.ply'.format(epoch)), p, n)
        if exp_mesh:
            dir_mesh = self.cfg['train']['dir_mesh']
            with torch.no_grad():
                v, f, _ = mc_from_psr(psr_grid,
                        zero_level=self.cfg['data']['zero_level'], real_scale=True, mask=torch.where(self.pcv==1, True, False).cpu().numpy())
                v = v * 2 - 1
                if scale is not None:
                    v *= scale
                if center is not None:
                    v += center
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(v)
            mesh.triangles = o3d.utility.Vector3iVector(f)
            outdir_mesh = os.path.join(dir_mesh, '{:04d}.ply'.format(epoch))
            o3d.io.write_triangle_mesh(outdir_mesh, mesh)

        if self.cfg['train']['vis_psr']:
            dir_psr_vis = self.cfg['train']['out_dir']+'/psr_vis_all'
            visualize_psr_grid(psr_grid, out_dir=dir_psr_vis)

    def gen_envelope(self, data):
        #--------------------------------------------
        vertices = data['target_points'].clone().detach().cpu().numpy()[0]
        # print(vertices.shape, vertices.mean(), vertices.max(), vertices.min(), np.max(np.max(np.abs(vertices), axis=0)))

        vertices = vertices * (self.cfg['model']['grid_res']/2)
        vertices = vertices + np.array([
            self.cfg['model']['grid_res']/2,
            self.cfg['model']['grid_res']/2,
            self.cfg['model']['grid_res']/2
        ])
        xy = vertices[:,[1,2]].astype(np.int32)
        z = vertices[:,0]

        voxels=np.zeros((self.cfg['model']['grid_res']+1,self.cfg['model']['grid_res'],self.cfg['model']['grid_res']))

        for i in range(len(z)):
            zc = int(np.ceil(z[i]))
            zf = int(np.floor(z[i]))
            voxels[zc][tuple(xy[i])]=1
            voxels[zf][tuple(xy[i])]=1

        i=0
        j=self.cfg['model']['grid_res']
        while voxels[i].sum()==0:
            i+=1
        while voxels[j].sum()==0:
            j-=1

        flag=True
        while i<=j:
            if voxels[i].sum()==0:
                voxels[i]=voxels[i-1]

            if voxels[j].sum()==0:
                voxels[j]=voxels[j+1]
            
            i+=1
            j-=1

        voxels2=np.zeros((self.cfg['model']['grid_res'],self.cfg['model']['grid_res'],self.cfg['model']['grid_res']))
        
        for i in range(self.cfg['model']['grid_res']):
            voxels2[i] = cv2.dilate(voxels[i], np.ones((3,3), np.float32))

            # img = (voxels[i]*255).astype(np.uint8)
            # img = cv2.ximgproc.thinning(img)
            # # plt.imshow(img, cmap=plt.cm.gray)
            # # plt.show()
            # pnts = cv2.findNonZero(img)
            # if pnts is not None:
            #     pnts = np.squeeze(pnts)
            #     ext = self.get_end_pnts(pnts, img)
            #     for p in ext:
            #         voxels2[i][p[0]-2:p[0]+3,p[1]-2:p[1]+3] = 0.0
            
        self.pcv = torch.tensor(voxels2).to(self.device)

        for i in range(self.cfg['model']['grid_res']):
            plt.imsave(f'out/x/{i}_.png', voxels2[i], cmap=plt.cm.gray)
        #=================================================

    # def get_end_pnts(self, pnts, img):
    #     extremes = []    
    #     for p in pnts:
    #         try:
    #             x = p[0]
    #             y = p[1]
    #             n = 0        
    #             n += img[y - 1,x]
    #             n += img[y - 1,x - 1]
    #             n += img[y - 1,x + 1]
    #             n += img[y,x - 1]    
    #             n += img[y,x + 1]    
    #             n += img[y + 1,x]    
    #             n += img[y + 1,x - 1]
    #             n += img[y + 1,x + 1]
    #             n /= 255        
    #             if n == 1:
    #                 extremes.append(p)
    #         except:
    #             extremes.append(p)
    #     return extremes
