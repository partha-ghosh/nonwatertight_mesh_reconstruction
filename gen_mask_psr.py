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

purpose = 'train'
# purpose = 'test'

os.system(f'mkdir -p dat/{purpose}/pointclouds && rm -rf dat/{purpose}/pointclouds/*')
os.system(f'mkdir -p dat/{purpose}/psrs && rm -rf dat/{purpose}/psrs/*')
os.system(f'mkdir -p dat/{purpose}/masks && rm -rf dat/{purpose}/masks/*')

# os.system(f"mkdir -p dat/unit_meshes && rm -rf dat/unit_meshes/*")

for _, _, f in os.walk(f'dat/{purpose}/meshes'):
    mesh_files = f

count = 0

for mesh_file in mesh_files:
    print(f'processing {mesh_file} ...')
    try:
        mesh = o3d.io.read_triangle_mesh(f"dat/{purpose}/meshes/{mesh_file}")
        
        orientations = [(0,0,0)]
        for i in range(5):
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
            # o3d.io.write_triangle_mesh(f'dat/unit_meshes/{count}.obj', mesh_copy2)
            pointcloud = mesh_copy2.sample_points_uniformly(number_of_points=1000000)
            pointcloud.estimate_normals()
            points = torch.tensor(np.asarray(pointcloud.points), dtype=torch.float32).to(device)[None,:]
            normals = torch.tensor(np.asarray(pointcloud.normals), dtype=torch.float32).to(device)[None,:]
            psr_grid = dpsr(points, normals).cpu().numpy()[0]

            # mesh_copy2.translate(np.ones(3)*(-0.5))
            pointcloud = mesh_copy2.sample_points_uniformly(number_of_points=500000)

            #various random trimming
            pcd = np.asarray(pointcloud.points)
            r = np.random.rand()/4
            l = np.random.rand()
            h = l + r 
            pcd = pcd[(pcd[:,2]<l) | (pcd[:,2]>h)] if np.random.randint(0,2) else pcd
            r = np.random.rand()/4
            l = np.random.rand()
            h = l + r 
            pcd = pcd[(pcd[:,1]<l) | (pcd[:,1]>h)] if np.random.randint(0,2) else pcd
            r = np.random.rand()/4
            l = np.random.rand()
            h = l + r 
            pcd = pcd[(pcd[:,0]<l) | (pcd[:,0]>h)] if np.random.randint(0,2) else pcd
            
            pcd_grid = copy.deepcopy(pcd) * voxel_grid_size
            pcd_grid = np.asarray(pcd_grid, dtype=int)
            mask = np.zeros((voxel_grid_size, voxel_grid_size, voxel_grid_size))
            k1,k2 = 3,4
            for x,y,z in pcd_grid:
                mask[x-k1:x+k2, y-k1:y+k2, z-k1:z+k2] = 1
            if len(pcd) >= 10000 and (random.choice([0,0,0,1]) if len(pcd) == len(pointcloud.points) else True):
                np.save(f"dat/{purpose}/psrs/{count}.npy", psr_grid)
                np.save(f"dat/{purpose}/masks/{count}.npy", mask)
                np.save(f"dat/{purpose}/pointclouds/{count}.npy", pcd)
                count += 1
    except:
        pass
        