import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import copy

voxel_grid_size = 128

os.system('mkdir -p data/train/pointclouds')
os.system('mkdir -p data/train/masks')

os.system('mkdir -p data/test/pointclouds')
os.system('mkdir -p data/test/masks')

os.system('mkdir -p vis')
os.system('mkdir -p vis2')

# purpose = 'train'
purpose = 'test'
for _, _, f in os.walk(f'data/{purpose}/meshes'):
    mesh_files = f

count = 0

for mesh_file in mesh_files:
    mesh = o3d.io.read_triangle_mesh(f"data/{purpose}/meshes/{mesh_file}")
    
    for _ in range(10):
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.rotate(o3d.geometry.get_rotation_matrix_from_xyz(np.random.random(3) * np.pi), mesh_copy.get_center())
        mesh_copy.translate(-mesh_copy.get_min_bound())
        mesh_copy.scale(0.9*voxel_grid_size/max(mesh_copy.get_max_bound()), (0,0,0))
        delta = 0.1 * voxel_grid_size / 2
        mesh_copy.translate(np.ones(3)*delta)

        pointcloud = np.zeros((voxel_grid_size, voxel_grid_size, voxel_grid_size))
        mask = np.zeros((voxel_grid_size, voxel_grid_size, voxel_grid_size))

        pcd = mesh_copy.sample_points_uniformly(number_of_points=10000)
        pcd = np.asarray(pcd.points)
        pcd = pcd + np.random.randn(10000, 3)
        pcd = np.asarray(pcd, dtype=int)
        
        for x,y,z in pcd:
            pointcloud[x,y,z] += 1

        pcd = mesh_copy.sample_points_uniformly(number_of_points=100000)
        pcd = np.asarray(pcd.points, dtype=int)
        for x,y,z in pcd:
            mask[x-2:x+3, y-2:y+3, z-2:z+3] = 1

        np.save(f"data/{purpose}/pointclouds/{count}.npy", pointcloud)
        np.save(f"data/{purpose}/masks/{count}.npy", mask)
        count += 1
        