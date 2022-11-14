import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from pytorch3d.loss import chamfer_distance
from skimage import measure
import open3d as o3d
from src.model import PSR2Mesh
from model import UNET

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

class MaskDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        for _, _, filenames in os.walk(f'{root_dir}/psrs'):
            self.filenames = filenames
        
    def __getitem__(self, index):
        psr = np.load(f'{self.root_dir}/psrs/{self.filenames[index]}')
        pcd = np.load(f'{self.root_dir}/pointclouds/{self.filenames[index]}') 
        
        return \
            torch.tensor(psr, dtype=torch.float)[None,:],\
            torch.tensor(pcd, dtype=torch.float)

    def __len__(self):
        return len(self.filenames)

# train_dataset = MaskDataset('data/train')
# print(train_dataset[0][1].shape)
# exit()

# hyperparameters
num_epochs = 1000
batch_size = 1
learning_rate = 5e-4

model = UNET(1, 1, features=[8, 16, 32, 64, 128]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = chamfer_distance

try:
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
except:
    pass

# dataset
train_dataset = MaskDataset('./data/train')
test_dataset = MaskDataset('./data/test')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

test_loss_history = []
loss_history = []
rolling_loss = 0
old_loss = 1e100

psr2mesh = PSR2Mesh.apply

# training loop
for epoch in range(num_epochs):
    for i, (psrs, pcds) in enumerate(train_loader):
        psrs = psrs.to(device)
        pcds = pcds.to(device)
        
        predicted_mask = model(psrs)

        predicted_pcds = list()
        for i in range(batch_size):
            psr = psrs[i,0][None, :]
            PSR2Mesh.use_mask(np.where(predicted_mask[i,0].cpu()>0.5, True, False))
            v, f, n = PSR2Mesh.apply(psr)
            predicted_pcds.append(v)

        # print(psrs.shape, masks.shape, outputs.shape)
        loss = 0
        for i in range(batch_size):
            loss += criterion(pcds[i][None,:], predicted_pcds[i])[0]
        
        rolling_loss = 0.975*rolling_loss + 0.025*loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        # import subprocess as sp
        # import os

        # def get_gpu_memory():
        #     command = "nvidia-smi --query-gpu=memory.free --format=csv"
        #     memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        #     memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        #     return memory_free_values

        # print(get_gpu_memory())

        if (i+1)%1 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item():.4f}, rolling_loss = {rolling_loss:.4f}')
    
    if epoch % 2 == 0:
        if loss.item() < old_loss:
            torch.save(model.state_dict(), f'model.pth')
            old_loss = rolling_loss

        with torch.no_grad():
            psr, pcd = next(iter(test_loader))
            psr = psr.to(device)
            predicted_mask = model(psr).cpu()
            
            verts, faces, _, _ = measure.marching_cubes(pcd[0].cpu().numpy(), mask=predicted_mask[0,0])
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d.io.write_triangle_mesh('predicted_mesh.ply', mesh)

            for i in range(predicted_mask.shape[-1]):
                plt.imsave(f"vis2/{i}op.png", psr[0,0,:,:,i], cmap=plt.cm.gray)  
                plt.imsave(f"vis2/{i}pm.png", predicted_mask[0,0,:,:,i], cmap=plt.cm.gray)  

        # loss_history = loss_history[-5000:] 
        # plt.plot(loss_history)
        # plt.savefig('loss.png')
        # plt.show()
                
        plt.figure()
        plt.plot(loss_history)
        plt.savefig('train_loss.png')
                    
        plt.figure()
        plt.plot(test_loss_history)
        plt.savefig('test_loss.png')
