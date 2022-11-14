import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import copy
from src.dpsr import DPSR
import torch
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

voxel_grid_size = 128

dpsr = DPSR(res=(voxel_grid_size,voxel_grid_size,voxel_grid_size), sig=2.0 if voxel_grid_size<=128 else 3.0).to(device)

os.system('mkdir -p dat/train/pointclouds')
os.system('mkdir -p dat/train/psrs')
os.system('mkdir -p dat/train/masks')

os.system('mkdir -p dat/test/pointclouds')
os.system('mkdir -p dat/test/psrs')
os.system('mkdir -p dat/test/masks')

os.system('mkdir -p vis')

purpose = 'train'
# purpose = 'test'
for _, _, f in os.walk(f'dat/{purpose}/meshes'):
    mesh_files = f

count = 0

for mesh_file in mesh_files:
    mesh = o3d.io.read_triangle_mesh(f"dat/{purpose}/meshes/{mesh_file}")
    
    orientations = [(0,0,0)]
    for i in range(9):
        orientations.append(np.random.random(3) * np.pi)

    for orientation in orientations:
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.rotate(o3d.geometry.get_rotation_matrix_from_xyz(orientation), mesh_copy.get_center())
        mesh_copy.translate(-mesh_copy.get_min_bound())
        mesh_copy.scale(0.9*voxel_grid_size/max(mesh_copy.get_max_bound()), (0,0,0))
        delta = 0.1 * voxel_grid_size / 2
        mesh_copy.translate(np.ones(3)*delta)

        mesh_copy2 = copy.deepcopy(mesh_copy)
        mesh_copy2.scale(1/voxel_grid_size, (0,0,0))
        pointcloud = mesh_copy2.sample_points_uniformly(number_of_points=1000000)
        pointcloud.estimate_normals()
        points = torch.tensor(np.asarray(pointcloud.points), dtype=torch.float32).to(device)[None,:]
        normals = torch.tensor(np.asarray(pointcloud.normals), dtype=torch.float32).to(device)[None,:]
        psr_grid = dpsr(points, normals).cpu().numpy()[0]

        # mesh_copy2.translate(np.ones(3)*(-0.5))
        pointcloud = mesh_copy2.sample_points_uniformly(number_of_points=10000)

        #various random trimming
        x = pointcloud.points
        x = x[(x[:,2]<0.4) & (x[:,2]>0.6)] if np.random.randint(0,2) else x
        x = x[(x[:,1]<0.4) & (x[:,1]>0.6)] if np.random.randint(0,2) else x
        x = x[(x[:,0]<0.4) & (x[:,0]>0.6)] if np.random.randint(0,2) else x

        mask = np.zeros((voxel_grid_size, voxel_grid_size, voxel_grid_size))
        pcd = mesh_copy.sample_points_uniformly(number_of_points=100000)
        pcd = np.asarray(pcd.points, dtype=int)
        k1,k2 = 3,4
        for x,y,z in pcd:
            mask[x-k1:x+k2, y-k1:y+k2, z-k1:z+k2] = 1

        np.save(f"dat/{purpose}/psrs/{count}.npy", psr_grid)
        np.save(f"dat/{purpose}/masks/{count}.npy", mask)
        np.save(f"dat/{purpose}/pointclouds/{count}.npy", pointcloud.points)
        count += 1
        