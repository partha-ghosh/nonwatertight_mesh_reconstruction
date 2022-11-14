import matplotlib.pyplot as plt
import numpy as np
import os

os.system('mkdir -p vis && rm -rf vis/*')

purpose = 'test' 

for n in range(10):
    # pcd = np.load(f'dat/{purpose}/pointclouds/{n}.npy')
    psr = np.load(f'dat/{purpose}/psrs/{n}.npy')
    mask = np.load(f'dat/{purpose}/masks/{n}.npy')
    os.system(f"mkdir -p vis/{n}")
    for i in range(psr.shape[2]):
        plt.imsave(f"vis/{n}/{i}psr.png", psr[:,:,i], cmap=plt.cm.gray)
        plt.imsave(f"vis/{n}/{i}mask.png", mask[:,:,i], cmap=plt.cm.gray)
