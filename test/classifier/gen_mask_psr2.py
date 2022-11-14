import matplotlib.pyplot as plt
import numpy as np
import trimesh
import os
import copy
from src.dpsr import DPSR
import torch
import random
import open3d as o3d

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
import sys

def visualize_psr_grid(psr_grid, pose=None, out_dir=None, out_video_name='video.mp4'):
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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

voxel_grid_size = 128

dpsr = DPSR(res=(voxel_grid_size,voxel_grid_size,voxel_grid_size), sig=2.0 if voxel_grid_size<=128 else 3.0).to(device)

# purpose = 'train'
# purpose = 'test'

purpose = sys.argv[1]

os.system(f'mkdir -p data/{purpose}/pointclouds && rm -rf data/{purpose}/pointclouds/*')
os.system(f'mkdir -p data/{purpose}/pointclouds_nonwatertight && rm -rf data/{purpose}/pointclouds_nonwatertight/*')
# os.system(f'mkdir -p dat/{purpose}/psrs && rm -rf dat/{purpose}/psrs/*')
# os.system(f'mkdir -p dat/{purpose}/masks && rm -rf dat/{purpose}/masks/*')

# os.system(f"mkdir -p dat/unit_meshes && rm -rf dat/unit_meshes/*")

for _, _, f in os.walk(f'data/{purpose}/meshes'):
    mesh_files = f

count = 0
for mesh_file in mesh_files:
    print(f'processing {mesh_file} ...')

    mesh0 = o3d.io.read_triangle_mesh(f"data/{purpose}/meshes/{mesh_file}")
        
    orientations = [(0,0,0)]
    for i in range(5):
        orientations.append(np.random.random(3) * np.pi)

    for orientation in orientations:
        mesh_copy = copy.deepcopy(mesh0)
        mesh_copy.rotate(o3d.geometry.get_rotation_matrix_from_xyz(orientation), mesh_copy.get_center())
        mesh_copy.translate(-mesh_copy.get_min_bound())
        mesh_copy.scale(0.9/max(mesh_copy.get_max_bound()), (0,0,0))
        delta = 0.1 / 2
        mesh_copy.translate(np.ones(3)*delta)
        o3d.io.write_triangle_mesh("/tmp/x.obj", mesh_copy)

        mesh = trimesh.load(f"/tmp/x.obj")
        
        n_points = 3000
        points, faces = mesh.sample(n_points, return_index=True)
        # normals = mesh.face_normals[faces]
        # points = (points/1.2) + 0.5

        # psr_grid = dpsr(
        #     torch.tensor(np.asarray(points), dtype=torch.float32).to(device)[None,:],
        #     torch.tensor(np.asarray(normals), dtype=torch.float32).to(device)[None,:]).cpu()[0]
        # visualize_psr_grid(psr_grid, out_dir='vis')
        # exit()

        flag = True
        while flag:
            #various random trimming
            pcd = copy.deepcopy(points)
            n_crop = np.random.randint(1,4)
            for _ in range(n_crop):
                axis = np.random.randint(0,3)
                choice = np.random.randint(0,3)
                if np.random.randint(0,2) == 0:
                    l, h = 0, 0
                    while h-l<0.15:
                        l, h = np.random.rand(), np.random.rand()
                        l, h = min(l,h), max(l,h)
                    pcd = pcd[(pcd[:,axis]<l) | (pcd[:,axis]>h)]
                else:
                    l, h = 0, 0
                    while h-l<0.33:
                        l, h = np.random.rand(), np.random.rand()
                        l, h = min(l,h), max(l,h)
                    pcd = pcd[(pcd[:,axis]>l) | (pcd[:,axis]<h)]
                
            # pcd_grid = copy.deepcopy(pcd) * voxel_grid_size
            # pcd_grid = np.asarray(pcd_grid, dtype=int)
            # mask = np.zeros((voxel_grid_size, voxel_grid_size, voxel_grid_size))
            # k1,k2 = 3, 4
            # for x,y,z in pcd_grid:
            #     mask[x-k1:x+k2, y-k1:y+k2, z-k1:z+k2] = 1
            if len(pcd) >= n_points/5:
                if len(pcd) != len(points):
                    np.save(f"data/{purpose}/pointclouds/{count}.npy", points)
                    np.save(f"data/{purpose}/pointclouds_nonwatertight/{count}.npy", pcd)
                    # trimesh.points.PointCloud(pcd).export(f'data/{purpose}/pointclouds_nonwatertight/{count}.obj')
                    # trimesh.points.PointCloud(points).export(f'data/{purpose}/pointclouds/{count}.obj')
                    flag = False

        count += 1
