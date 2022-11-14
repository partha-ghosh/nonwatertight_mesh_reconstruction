import numpy as np
import matplotlib.pyplot as plt

with open('3d_data/90276.obj', 'r') as f:
    f = f.readlines()

points = np.array([eval(f[i][2:].replace(' ',',')) for i in range(4, len(f)) if f[i][:2]=='v '])
center = (np.max(points, axis=0)+np.min(points, axis=0))/2
points=points-center
M = np.max(abs(points))

points = (points*120/M)+128
points = points + np.random.randn(*points.shape)
xy = points[:,:2].astype(np.int32)
z = points[:,2]
del points

voxels=np.zeros((257,256,256))

for i in range(len(z)):
    zc = int(np.ceil(z[i]))
    zf = int(np.floor(z[i]))
    voxels[zc][tuple(xy[i])]=1
    voxels[zf][tuple(xy[i])]=1

i=0
j=256
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

for i in range(256):
    plt.imsave(f'3d_data/x/{i}.png', voxels[i], cmap=plt.cm.gray)