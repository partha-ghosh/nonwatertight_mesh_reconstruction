
import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt

mesh = o3d.io.read_triangle_mesh(f"data/thingi/x.obj")
mesh.translate(-mesh.get_min_bound())
mesh.scale(0.9/max(mesh.get_max_bound()), (0,0,0))
delta = 0.1 / 2
mesh.translate(np.ones(3)*delta)
# mesh.translate(np.ones(3)*(-0.5))


pointcloud = mesh.sample_points_uniformly(number_of_points=100000)
pointcloud.estimate_normals()

points = np.asarray(pointcloud.points)
normals = np.asarray(pointcloud.normals)

point_density = []
for i in range(20):
    random_point = points[np.random.randint(0,len(points),1)[0]]
    norms = np.linalg.norm(points - random_point,axis=1)
    norms = norms[np.where(norms<3*3**0.5/128, True, False)]
    point_density.append(len(norms))
print(np.mean(point_density))

os.system('mkdir -p data/demo/custom/0')

with open('data/demo/custom/test.lst', 'w') as f:
    f.write('0')

np.savez('data/demo/custom/0/pointcloud.npz', 
        points=points,
        normals=normals)