import torch
import time
import trimesh
import numpy as np
from src.utils import mc_from_psr
import matplotlib.pyplot as plt
import cv2
from src.model import UNET2D, UNET3D
import os
from patchify import patchify, unpatchify
from scipy import ndimage
from tqdm import trange
from src.utils import load_model_manual

class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        device (device): pytorch device
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, device=None, padding=0.1, 
                 sample=False, input_type = None, dpsr=None, psr_tanh=True):
        self.model = model.to(device)
        
        state_dict = torch.load('/home/scholar/tmp/shape_as_points/out/demo_shapenet_outlier/model.pt')
        load_model_manual(state_dict['state_dict'], self.model)
        self.model.eval()


        self.points_batch_size = points_batch_size
        self.threshold = threshold
        self.device = device
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.dpsr = dpsr
        self.psr_tanh = psr_tanh
        #os.system("mkdir -p out/vis && rm -rf out/vis/*")
        
    def generate_mesh(self, data, return_stats=True, cfg=None):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''

        self.model.eval()
        stats_dict = {}

        t0 = time.time()

        mesh = trimesh.load(f"/home/scholar/tmp/sap_test/3d/data/train/meshes/plane.obj")
        n_points = 30000
        p, faces = mesh.sample(n_points, return_index=True)
        normals = mesh.face_normals[faces]
        normals = torch.tensor(normals[None,:], dtype=torch.float).to(self.device)

        p = (0.5*p/1.2) + 0.5
        
        # p = data.get('inputs', torch.empty(1, 0)).to(self.device) 
        
        # p = np.load(f'dat/test/pointclouds/0.npy')
        # p = p[np.random.randint(0, len(p), 3000)]
        p = p + 0.005 * np.random.randn(*p.shape)
        # n_outliers = int(len(p) * 0.25)
        # outlier_indices = np.random.randint(0, len(p), n_outliers)
        # p[outlier_indices] = np.random.uniform(0, 1, (n_outliers, 3))
        points = torch.tensor(p[None,:], dtype=torch.float).to(self.device)
        # points, normals = self.model(p)
        
       
        # points = torch.tensor(np.load('data/demo/custom/0/pointcloud.npz')['points']).to(self.device)[None,:]
        # normals = torch.tensor(np.load('data/demo/custom/0/pointcloud.npz')['normals']).to(self.device)[None,:]
        
        t1 = time.time()
        psr_grid = self.dpsr(points, normals)
        # self.visualize_psr_grid(psr_grid.cpu(), out_dir='vis')

        # self.gen_mask_with_dilation(points)
        # self.gen_mask_with_2d_unet(points)
        # self.gen_mask_with_3d_unet_from_pcd(points)
        self.gen_mask_with_3d_unet_from_psr(psr_grid)

        s = psr_grid.shape[-1] # size of psr_grid
        # for i in range(s):
        #     plt.imsave(f'out/vis/{i}mask_psr.png', psr_grid[0][i].clone().detach().cpu().numpy()+self.mask[i], cmap=plt.cm.gray)
        #     plt.imsave(f'out/vis/{i}psr.png', psr_grid[0][i].clone().detach().cpu().numpy(), cmap=plt.cm.gray)
        #     plt.imsave(f'out/vis/{i}mask.png', self.mask[i], cmap=plt.cm.gray)
            # plt.imsave(f'out/x/{i}.png', cv2.filter2D(psr_grid[0][i].clone().detach().cpu(), -1, kernel)+self.pcv[i].cpu(), cmap=plt.cm.gray)
        # input("Done >>>")

        t2 = time.time()
        v, f, _ = mc_from_psr(psr_grid, 
                    zero_level=self.threshold,)
                    #mask=np.where(self.mask>0.6, True, False))
        stats_dict['pcl'] = t1 - t0
        stats_dict['dpsr'] = t2 - t1
        stats_dict['mc'] = time.time() - t2
        stats_dict['total'] = time.time() - t0

        v, f = self.trim_mesh_using_mask(v, f, np.where(self.mask>0.5, True, False))

        if return_stats:
            return v, f, points, normals, stats_dict
        else:
            return v, f, points, normals


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
        
    def visualize_psr_grid(self, psr_grid, pose=None, out_dir=None, out_video_name='video.mp4'):
        if pose is not None:
            device = psr_grid.device
            # get world coordinate of grid points [-1, 1]
            res = psr_grid.shape[-1]
            x = torch.linspace(-1, 1, steps=res)
            co_x, co_y, co_z = torch.meshgrid(x, x, x)
            co_grid = torch.stack(
                    [co_x.reshape(-1), co_y.reshape(-1), co_z.reshape(-1)],
                    dim=1).to(device).unsqueeze(0)

            # visualize the projected occ_soft value
            res = 128
            psr_grid = psr_grid.reshape(-1)
            out_mask = psr_grid>0
            in_mask = psr_grid<0
            pix = pose.transform_points_screen(co_grid, ((res, res),))[..., :2].round().long().squeeze()
            vis_mask = (pix[..., 0]>=0) & (pix[..., 0]<=res-1) & \
                        (pix[..., 1]>=0) & (pix[..., 1]<=res-1)
            pix_out = pix[vis_mask & out_mask]
            pix_in = pix[vis_mask & in_mask]

            img = torch.ones([res,res]).to(device)
            psr_grid = torch.sigmoid(- psr_grid * 5)
            img[pix_out[:, 1], pix_out[:, 0]] = psr_grid[vis_mask & out_mask]
            img[pix_in[:, 1], pix_in[:, 0]] = psr_grid[vis_mask & in_mask]
            # save_image(img, 'tmp.png', normalize=True)
            return img
        elif out_dir is not None:
            dir_psr_vis = out_dir
            os.makedirs(dir_psr_vis, exist_ok=True)
            psr_grid = psr_grid.squeeze().detach().cpu().numpy()
            axis = ['x', 'y', 'z']
            s = psr_grid.shape[0]
            for idx in trange(s):
                my_dpi = 100
                plt.figure(figsize=(1000/my_dpi, 300/my_dpi), dpi=my_dpi)
                plt.subplot(1, 3, 1)
                plt.imshow(ndimage.rotate(psr_grid[idx], 180, mode='nearest'), cmap='nipy_spectral')
                plt.clim(-1, 1)
                plt.colorbar()
                plt.title('x')
                plt.grid("off")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(ndimage.rotate(psr_grid[:, idx], 180, mode='nearest'), cmap='nipy_spectral')
                plt.clim(-1, 1)
                plt.colorbar()
                plt.title('y')
                plt.grid("off")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(ndimage.rotate(psr_grid[:,:,idx], 90, mode='nearest'), cmap='nipy_spectral')
                plt.clim(-1, 1)
                plt.colorbar()
                plt.title('z')
                plt.grid("off")
                plt.axis("off")


                plt.savefig(os.path.join(dir_psr_vis, '{}'.format(idx)), pad_inches = 0, dpi=100)


    def gen_mask_with_dilation(self, points):
        voxel_grid = self.gen_voxel_grid(points)

        for i in range(self.dpsr.res[0]):
            voxel_grid[i] = cv2.GaussianBlur(voxel_grid[i], (5,5), 0)
            voxel_grid[i] = np.where(voxel_grid[i]>(voxel_grid[i].max()+voxel_grid[i].min())*3/10, 1.0, 0.0)
            voxel_grid[i] = cv2.dilate(voxel_grid[i], np.ones((9, 9), np.float32))
        
        self.mask = np.asarray(voxel_grid)

    def gen_mask_with_2d_unet(self, points):
        # voxels2=np.zeros((self.dpsr.res[0],self.dpsr.res[0],self.dpsr.res[0]))

        # for i in range(self.dpsr.res[0]):
        #     voxels2[i] = np.where(voxels[i]/voxels[i].max()<0.3, 0.0, 1.0)

        # print(np.array([np.count_nonzero(voxels2[i]) for i in range(self.dpsr.res[0])]).mean())

        # for i in range(self.dpsr.res[0]):
        #     img = voxels[i-2:i+3].sum(0)
        #     img = img/img.max()
        #     voxels2[i] = np.where(img>0.03, 1, 0)

        model = UNET2D(1, 1, features=[16, 32, 64, 128, 256, 512]).to(self.device)
        try:
            model.load_state_dict(torch.load('unet2d.pkl')['model_state'])
            model.eval()
            print("model loaded")
        except:
          pass

        voxel_grid = self.gen_voxel_grid(points)

        with torch.no_grad():
            cross_sections_pc = voxel_grid[:,None,:,:]
            cross_sections_pc = torch.from_numpy(cross_sections_pc).type(torch.float).to(self.device)
            
            cross_sections = torch.zeros((self.dpsr.res[0],1,self.dpsr.res[0],self.dpsr.res[0]))
            for i in range(0,self.dpsr.res[0],8):
                cross_sections[i:i+8] = model(cross_sections_pc[i:i+8]).cpu()
    
        self.mask = np.asarray(cross_sections.reshape(self.dpsr.res[0],self.dpsr.res[0],self.dpsr.res[0]))


    def gen_mask_with_3d_unet_from_pcd(self, points):
        model = UNET3D(1, 1, features=[8,16,32,64,128]).to(self.device)
        try:
            model.load_state_dict(torch.load('unet3d.pth'))
            model.eval()
            print("model loaded")
        except:
          pass

        voxel_grid = self.gen_voxel_grid(points)

        for i in range(self.dpsr.res[0]):
            plt.imsave(f'out/vis/{i}pcd.png', voxel_grid[i], cmap=plt.cm.gray)

        patch_size = 64
        voxel_patches = patchify(voxel_grid, (patch_size,patch_size,patch_size), step=patch_size)
        patch_orientation = np.array(voxel_patches.shape[:3])
        num_patches = patch_orientation.cumprod()[-1]

        mask_patches = torch.zeros(voxel_patches.shape).to(self.device)

        with torch.no_grad():
            for i in range(num_patches):
                voxel_patch = voxel_patches[np.unravel_index(i, patch_orientation)][None, None, :]
                voxel_patch = torch.tensor(voxel_patch, dtype=torch.float).to(self.device)
                
                mask_patch = model(voxel_patch)
                mask_patches[np.unravel_index(i, patch_orientation)] = mask_patch[0,0]
        
        self.mask = unpatchify(mask_patches.cpu().numpy(), voxel_grid.shape)

    def gen_mask_with_3d_unet_from_psr(self, psr_grid):
        model = UNET3D(1, 1, features=[8, 16, 32, 64, 128]).to(self.device)
        try:
            model.load_state_dict(torch.load('/home/scholar/tmp/shape_as_points/out/demo_shapenet_outlier/unet.pt'))
            model.eval()
            print("model loaded")
        except:
          pass

        # for i in range(self.dpsr.res[0]):
        #     plt.imsave(f'out/vis/{i}pcd.png', psr_grid[0][i].detach().cpu().numpy(), cmap=plt.cm.gray)

        with torch.no_grad():
            self.mask = model(torch.tensor(psr_grid[None,:], dtype=torch.float))[0][0].cpu().numpy()
            

    def gen_voxel_grid(self, points):

        vertices = points.clone().detach().cpu().numpy()[0]
        vertices = vertices * self.dpsr.res[0]
        vertices = vertices.astype(int)

        voxel_grid = np.zeros((self.dpsr.res[0],self.dpsr.res[0],self.dpsr.res[0]))
        for x,y,z in vertices: 
            voxel_grid[x,y,z] += 1

        voxel_grid = (voxel_grid-voxel_grid.mean())/voxel_grid.std()

        # vertices = points.clone().detach().cpu().numpy()[0]
        # # print(vertices.shape, vertices.mean(), vertices.max(), vertices.min(), np.max(np.max(np.abs(vertices), axis=0)))

        # vertices = vertices * self.dpsr.res[0]
        # xy = vertices[:,[1,2]].astype(np.int32)
        # z = vertices[:,0]

        # voxels=np.zeros((self.dpsr.res[0]+1,self.dpsr.res[0],self.dpsr.res[0]))

        # for i in range(len(z)):
        #     zc = int(np.ceil(z[i]))
        #     zf = int(np.floor(z[i]))
        #     voxels[zc][tuple(xy[i])]+=1
        #     voxels[zf][tuple(xy[i])]+=1

        # i=0
        # j=self.dpsr.res[0]
        # while voxels[i].sum()==0:
        #     i+=1
        # while voxels[j].sum()==0:
        #     j-=1

        # flag=True
        # while i<=j:
        #     if voxels[i].sum()==0:
        #         voxels[i]=voxels[i-1]

        #     if voxels[j].sum()==0:
        #         voxels[j]=voxels[j+1]
            
        #     i+=1
        #     j-=1

        # for i in range(self.dpsr.res[0]):
        #     voxels[i] = (np.clip((voxels[i]-voxels[i].mean())/voxels[i].std(), -3, 3) + 3)/6

        for i in range(self.dpsr.res[0]):
            plt.imsave(f'out/vis/{i}pcd.png', voxel_grid[i], cmap=plt.cm.gray)

        return voxel_grid
