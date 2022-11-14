import os
import numpy as np
import torch
from torch.nn import functional as F
from collections import defaultdict
import trimesh
from tqdm import tqdm

from src.dpsr import DPSR
from src.utils import grid_interp, export_pointcloud, export_mesh, \
                      mc_from_psr, scale2onet, GaussianSmoothing
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss import chamfer_distance
from pdb import set_trace as st
from scipy.spatial.distance import directed_hausdorff

import cv2 as cv
import matplotlib.pyplot as plt
from itertools import product

class Trainer(object):
    '''
    Args:
        model (nn.Module): our defined model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
    '''
    def __init__(self, cfg, optimizer, device=None):
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        if self.cfg['train']['w_raw'] != 0:
            from src.model import PSR2Mesh
            self.psr2mesh = PSR2Mesh.apply

        # initialize DPSR
        self.dpsr = DPSR(res=(cfg['model']['grid_res'], 
                            cfg['model']['grid_res'], 
                            cfg['model']['grid_res']), 
                        sig=cfg['model']['psr_sigma'])
        if torch.cuda.device_count() > 1:    
            self.dpsr = torch.nn.DataParallel(self.dpsr) # parallell DPSR
        self.dpsr = self.dpsr.to(device)

        if cfg['train']['gauss_weight']>0.:
            self.gauss_smooth = GaussianSmoothing(1, 7, 2).to(device)
        
    def train_step(self, inputs, data, model, unet):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        p = data['pointcloud'].to(self.device)
        
        out = model(p)
        
        points, normals = out

        loss = 0
        loss_each = {}
        if self.cfg['train']['w_psr'] != 0:
            psr_gt = data['psr'].to(self.device)
            # if self.cfg['model']['psr_tanh']:
            #     psr_gt = torch.tanh(psr_gt)
            
            psr_grid = self.dpsr(points, normals)
            # if self.cfg['model']['psr_tanh']:
            #     psr_grid = torch.tanh(psr_grid)

            # apply a rescaling weight based on GT SDF values
            if self.cfg['train']['gauss_weight']>0:
                gauss_sigma = self.cfg['train']['gauss_weight']
                # set up the weighting for loss, higher weights 
                # for points near to the surface
                psr_gt_pad = torch.nn.ReplicationPad3d(1)(psr_gt.unsqueeze(1)).squeeze(1)
                delta_x = delta_y = delta_z = 1
                grad_x = (psr_gt_pad[:, 2:, :, :] - psr_gt_pad[:, :-2, :, :]) / 2 / delta_x
                grad_y = (psr_gt_pad[:, :, 2:, :] - psr_gt_pad[:, :, :-2, :]) / 2 / delta_y
                grad_z = (psr_gt_pad[:, :, :, 2:] - psr_gt_pad[:, :, :, :-2]) / 2 / delta_z
                grad_x = grad_x[:, :, 1:-1, 1:-1]
                grad_y = grad_y[:, 1:-1, :, 1:-1]
                grad_z = grad_z[:, 1:-1, 1:-1, :]
                psr_grad = torch.stack([grad_x, grad_y, grad_z], dim=-1)
                psr_grad_norm = psr_grad.norm(dim=-1)[:, None]
                w = torch.nn.ReplicationPad3d(3)(psr_grad_norm)
                w = 2*self.gauss_smooth(w).squeeze(1)
                loss_each['psr'] = self.cfg['train']['w_psr'] * F.mse_loss(w*psr_grid, w*psr_gt)
            else:
                loss_each['psr'] = self.cfg['train']['w_psr'] * F.mse_loss(psr_grid, psr_gt)

            loss += loss_each['psr']

        # regularization on the input point positions via chamfer distance
        if self.cfg['train']['w_reg_point'] != 0.:
            points_gt = data.get('gt_points').to(self.device)
            loss_reg, loss_norm = chamfer_distance(points, points_gt)
                
            loss_each['reg'] = self.cfg['train']['w_reg_point'] * loss_reg
            loss += loss_each['reg']
            
        if self.cfg['train']['w_normals'] != 0.:
            points_gt = data.get('gt_points').to(self.device)
            normals_gt = data.get('gt_points.normals').to(self.device)
            x_nn = knn_points(points, points_gt, K=1)
            x_normals_near = knn_gather(normals_gt, x_nn.idx)[..., 0, :]
            
            cham_norm_x = F.l1_loss(normals, x_normals_near)
            loss_norm = cham_norm_x

            loss_each['normals'] = self.cfg['train']['w_normals'] * loss_norm
            loss += loss_each['normals']    
            
        if self.cfg['train']['w_raw'] != 0:
            res = self.cfg['model']['grid_res']
            # DPSR to get grid
            psr_grid = self.dpsr(points, normals)
            if self.cfg['model']['psr_tanh']:
                psr_grid = torch.tanh(psr_grid)
            
            v, f, n = self.psr2mesh(psr_grid)

            pts_gt = data.get('gt_points').to(self.device)
            
            loss, _ = chamfer_distance(v, pts_gt)


        mask_gt = data['mask'].to(self.device)
        predicted_mask = unet(psr_grid[:,None,...])
        # predicted_mask = unet(psr_gt[:,None,...])
        dice_loss = self.dice_loss(predicted_mask, mask_gt)
        loss_each['dice'] = dice_loss
        loss += dice_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_each

    # def dice_loss(self, predicted_mask, predicted_psr, target_surface):
    #     target_mask = torch.zeros_like(predicted_mask).to(device=self.device)
    #     for j in range(len(target_mask)):
    #         predicted_surface_indices = np.argwhere((-0.1<predicted_psr[j,0].cpu()) & (predicted_psr[j,0].cpu()<0.1))
    #         for i in range(len(predicted_surface_indices[0])):
    #             x = predicted_surface_indices[0][i]
    #             y = predicted_surface_indices[1][i]
    #             z = predicted_surface_indices[2][i]
    #             if torch.any(target_surface[x-4:x+5, y-4:y+5, z-4:z+5]==1):
    #                 target_mask[j,0][x-2:x+3, y-2:y+3, z-2:z+3] = 1
    #     dice_loss =  1 - \
    #         (1 + torch.sum(2 * target_mask * predicted_mask)) / \
    #         (1 + torch.sum(target_mask**2) + torch.sum(predicted_mask**2))
    #     return dice_loss 

    def dice_loss(self, predicted_mask, target_mask):
        dice_loss =  1 - \
            (1 + torch.sum(2 * target_mask * predicted_mask)) / \
            (1 + torch.sum(target_mask**2) + torch.sum(predicted_mask**2))
        return dice_loss


    # def tversky_loss(self, predicted, target):
    #     n_positive = torch.sum(target==1.0)
    #     beta = 1 - (n_positive / (np.cumprod(np.array(predicted.shape))[-1]))
    #     x = target * predicted
    #     x = (1 + torch.sum(x)) / (1 + torch.sum(x + beta * (predicted - x) + (1 - beta) * (target - x)))
    #     return 1 - x
    
    def trim_mesh_using_mask(self, vertices, faces, mask_3d):#, origin_min, origin_max):
        '''
        return:
            vertices_new
            faces_new
        '''	
        scale = mask_3d.shape[0]
        vertices = vertices * scale
        vertices_rasterized = vertices.astype(int)
        vertex_indices_at_mask = np.argwhere(mask_3d[vertices_rasterized[:,0], vertices_rasterized[:,1], vertices_rasterized[:,2]]).flatten()
        new_vertices = vertices[vertex_indices_at_mask]
        face_index_map = np.ones(len(vertices))*(-1)
        j = 0
        for i in range(len(vertices)):
            if (vertices[i] == new_vertices[j]).all(): 
                face_index_map[i] = j
                if j < len(new_vertices)-1:
                    j += 1
                else:
                    break

        new_faces = face_index_map[faces]
        new_faces = new_faces[(new_faces!=-1).all(axis=1)]
        new_vertices = new_vertices/scale
        
        return new_vertices, new_faces

    def save(self, model, unet, data, epoch, id):
        
        p = data['pointcloud'].to(self.device)

        exp_pcl = self.cfg['train']['exp_pcl']
        exp_mesh = self.cfg['train']['exp_mesh']
        exp_gt = self.cfg['generation']['exp_gt']
        exp_input = self.cfg['generation']['exp_input']
        
        model.eval()
        unet.eval()
        with torch.no_grad():
            points, normals = model(p)

        # if exp_gt:
        #     points_gt = data.get('gt_points').to(self.device)
        #     normals_gt = data.get('gt_points.normals').to(self.device)

        if exp_pcl:
            dir_pcl = self.cfg['train']['dir_pcl']
            export_pointcloud(os.path.join(dir_pcl, '{:04d}_{:01d}.ply'.format(epoch, id)), scale2onet(points), normals)
            # if exp_gt:
            #     export_pointcloud(os.path.join(dir_pcl, '{:04d}_{:01d}_oracle.ply'.format(epoch, id)), scale2onet(points_gt), normals_gt)
            if exp_input:
                export_pointcloud(os.path.join(dir_pcl, '{:04d}_{:01d}_input.ply'.format(epoch, id)), scale2onet(p))

        if exp_mesh:
            dir_mesh = self.cfg['train']['dir_mesh']
            psr_grid = self.dpsr(points, normals)
            with torch.no_grad():
                mask = unet(psr_grid[:,None,...])[0,0].cpu()
            
            for i in range(psr_grid.shape[2]):
                plt.imsave(f"out/vis/{i}psr_o.png", data['psr'][0,i,:,:], cmap=plt.cm.gray)
                plt.imsave(f"out/vis/{i}mask_o.png", data['mask'][0,0,i,:,:], cmap=plt.cm.gray)
                plt.imsave(f"out/vis/{i}psr_p.png", psr_grid[0,i,:,:].cpu(), cmap=plt.cm.gray)
                plt.imsave(f"out/vis/{i}mask_p.png", mask[i,:,:], cmap=plt.cm.gray)

            # psr_grid = torch.tanh(psr_grid)
            def exp_mesh(m, initials):
                with torch.no_grad():
                    v, f, _ = mc_from_psr(psr_grid, 
                                zero_level=self.cfg['data']['zero_level'], mask=np.where(m>0.5, True, False))
                    # v, f = self.trim_mesh_using_mask(v, f, np.where(mask>0.5, True, False))
                outdir_mesh = os.path.join(dir_mesh, '{}{:04d}_{:01d}.ply'.format(initials, epoch, id))
                export_mesh(outdir_mesh, scale2onet(v), f)

                cd = chamfer_distance(torch.tensor(v).reshape(1,-1,3), p.cpu())[0].item()
                hd = max(directed_hausdorff(v, p.cpu()[0])[0], directed_hausdorff(p.cpu()[0], v)[0])
                print(f'{initials} chamfer loss: {cd}')
                print(f'{initials} hausdorff distance: {hd}')
                with open('metric.csv', 'a') as f:
                    f.write(f'{initials}, {cd}, {hd}\n')

            exp_mesh(mask, f'predicted_mesh')

            # for i in np.linspace(0, 127, 11):
            #     mask_ = []
            #     for j in range(psr_grid.shape[2]):
            #         img = cv.imread(f'out/vis/{j}psr_p.png',0)
            #         laplacian = cv.Laplacian(img,cv.CV_64F)
            #         ret,th = cv.threshold(laplacian,i,255,cv.THRESH_BINARY)
            #         th = cv.dilate(th,np.ones((5,5),np.uint8),iterations = 1)
            #         plt.imsave(f"out/mask/{j}.png", th, cmap=plt.cm.gray)
            #         mask_.append(th/255)
            #     mask_ = np.array(mask_)
            #     if np.where(mask_>0.5, True, False).any():
            #         exp_mesh(mask_, f'2Dlaplacian_threshold{i}_mesh')

            # laplacian_kernel = np.array([
            #     [[0,0,0],[0,1,0],[0,0,0]],
            #     [[0,1,0],[1,-6,1],[0,1,0]],
            #     [[0,0,0],[0,1,0],[0,0,0]],
            # ])
            # psr_grid = psr_grid.cpu()
            # psr_grid2 = (psr_grid - psr_grid.min())
            # psr_grid2 = psr_grid2/psr_grid2.max()
            # for i in np.linspace(0, 127, 11):
            #     lap = np.zeros_like(psr_grid[0])
            #     mask_ = np.zeros_like(psr_grid[0])

            #     for x,y,z in product(range(len(lap)), range(len(lap)), range(len(lap))):
            #         if psr_grid2[0,x-1:x+2,y-1:y+2,z-1:z+2].shape==(3,3,3):
            #             lap[x,y,z] = torch.abs(torch.sum(psr_grid2[0,x-1:x+2,y-1:y+2,z-1:z+2]*laplacian_kernel)).item()

            #     lap = np.where(lap>i/255, 1, 0)

            #     k1,k2 = 2,3

            #     for x,y,z in product(range(len(lap)), range(len(lap)), range(len(lap))):
            #         if lap[x,y,z]>0:
            #             mask_[x-k1:x+k2, y-k1:y+k2, z-k1:z+k2] = 1.0
                
            #     for j in range(psr_grid.shape[2]):
            #         plt.imsave(f"out/mask/{j}.png", mask_[j,:,:])
            #     if mask_.any():
            #         exp_mesh(mask_, f'3Dlaplacian_threshold{i}_mesh')
            

            if exp_gt:
                psr_gt = data['psr']
                with torch.no_grad():
                    v, f, _ = mc_from_psr(psr_gt,
                            zero_level=self.cfg['data']['zero_level'])
                    v, f = self.trim_mesh_using_mask(v, f, np.where(mask>0.5, True, False))
                export_mesh(os.path.join(dir_mesh, '{:04d}_{:01d}_oracle.ply'.format(epoch, id)), scale2onet(v), f)
        

        
    def evaluate(self, val_loader, model, unet):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        eval_list = defaultdict(list)

        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step(data, model, unet)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict
    
    def eval_step(self, data, model, unet):
        ''' Performs an evaluation step.
        Args:
            data (dict): data dictionary
        '''
        model.eval()
        unet.eval()
        eval_dict = {}

        p = data['pointcloud'].to(self.device)
        psr_gt = data['psr'].to(self.device)
        mask_gt = data['mask'].to(self.device)
        
        with torch.no_grad():
            # forward pass
            points, normals = model(p)
            # DPSR to get predicted psr grid
            psr_grid = self.dpsr(points, normals)

            predicted_mask = unet(psr_grid[:,None,...])

        eval_dict['psr_l1'] = F.l1_loss(psr_grid, psr_gt).item()
        eval_dict['psr_l2'] = F.mse_loss(psr_grid, psr_gt).item()
        eval_dict['psr_l2+dice_loss'] = F.mse_loss(psr_grid, psr_gt).item() + self.dice_loss(predicted_mask, mask_gt).item()

        return eval_dict