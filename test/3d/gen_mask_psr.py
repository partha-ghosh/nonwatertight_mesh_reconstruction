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

os.system('mkdir -p vis')

# purpose = 'train'
purpose = 'test'

os.system(f'mkdir -p data/{purpose}/psrs && rm -rf data/{purpose}/psrs/*')
os.system(f'mkdir -p data/{purpose}/masks && rm -rf data/{purpose}/masks/*')
os.system(f'mkdir -p data/{purpose}/pointclouds && rm -rf data/{purpose}/pointclouds/*')

for _, _, f in os.walk(f'data/{purpose}/meshes'):
    mesh_files = f

count = 0

for mesh_file in mesh_files:
    mesh = o3d.io.read_triangle_mesh(f"data/{purpose}/meshes/{mesh_file}")
    
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
        pointcloud = mesh_copy2.sample_points_uniformly(number_of_points=100000)
        pointcloud.estimate_normals()
        points = torch.tensor(np.asarray(pointcloud.points), dtype=torch.float32).to(device)[None,:]
        normals = torch.tensor(np.asarray(pointcloud.normals), dtype=torch.float32).to(device)[None,:]
        psr_grid = dpsr(points, normals).cpu().numpy()[0]


        mask = np.zeros((voxel_grid_size, voxel_grid_size, voxel_grid_size))
        pcd = mesh_copy.sample_points_uniformly(number_of_points=10000)
        pcd = np.asarray(pcd.points, dtype=int)
        k1,k2 = 3,4
        for x,y,z in pcd:
            mask[x-k1:x+k2, y-k1:y+k2, z-k1:z+k2] = 1

        pcd_grid = np.zeros_like(mask)
        pcd = mesh_copy.sample_points_uniformly(number_of_points=3000)
        pcd = np.asarray(pcd.points, dtype=int)
        for x,y,z in pcd:
            pcd_grid[x,y,z] = 1

        np.save(f"data/{purpose}/psrs/{count}.npy", psr_grid)
        np.save(f"data/{purpose}/masks/{count}.npy", mask)
        np.save(f"data/{purpose}/pointclouds/{count}.npy", pcd_grid)
        # np.save(f"data/{purpose}/pointclouds/{count}.npy", np.asarray(pointcloud.points))
        count += 1
        