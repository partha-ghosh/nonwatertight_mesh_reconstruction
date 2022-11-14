import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
import numpy as np
import os
import random

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
            torch.tensor(pcd, dtype=torch.float)[None,:]

    def __len__(self):
        return len(self.filenames)

def balanced_cross_entropy_loss(predicted, target):
    n_positive = torch.sum(target==1.0)
    beta = 1 - (n_positive / (np.cumprod(np.array(predicted.shape))[-1]))
    ep = 10**-8
    return torch.sum(-beta*target*torch.log(predicted+ep)-(1-beta)*(1-target)*torch.log(1-predicted+ep))

def dice_loss(predicted, target):
    return 1 - \
        (1 + torch.sum(2 * target * predicted)) / \
        (1 + torch.sum(target**2) + torch.sum(predicted**2))


def tversky_loss(predicted, target):
    n_positive = torch.sum(target==1.0)
    beta = 1 - (n_positive / (np.cumprod(np.array(predicted.shape))[-1]))
    x = target * predicted
    x = (1 + torch.sum(x)) / (1 + torch.sum(x + beta * (predicted - x) + (1 - beta) * (target - x)))
    return 1 - x


# train_dataset = MaskDataset('data/train')
# print(train_dataset[0][0].shape)
# exit()

# hyperparameters
num_epochs = 100
batch_size = 8
learning_rate = 5e-4

model = UNET(1, 1, features=[8, 16, 32, 64, 128]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.L1Loss()
# criterion = balanced_cross_entropy_loss
# criterion = tversky_loss
# criterion = dice_loss

try:
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
except:
    pass

# dataset
train_dataset = MaskDataset('./data/train')
test_dataset = MaskDataset('./data/test')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)

test_loss_history = []
loss_history = []
rolling_loss = 0
old_loss = 9999

# training loop
for epoch in range(num_epochs):
    for i, (psrs, pcds) in enumerate(train_loader):
        psrs = psrs.to(device)
        pcds = pcds.to(device)

        outputs = model(pcds)
        # print(psrs.shape, masks.shape, outputs.shape)
        loss = criterion(outputs, psrs)
        
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
            pcd = pcd.to(device)
            
            predicted_psr = model(pcd).cpu()

            pcd = pcd.cpu()
            test_loss = criterion(predicted_psr, psr)
            test_loss_history.append(test_loss)

            for i in range(predicted_psr.shape[-1]):
                plt.imsave(f"vis2/{i}opcd.png", pcd[0,0,:,:,i], cmap=plt.cm.gray)
                plt.imsave(f"vis2/{i}opsr.png", psr[0,0,:,:,i], cmap=plt.cm.gray)  
                plt.imsave(f"vis2/{i}ppsr.png", predicted_psr[0,0,:,:,i], cmap=plt.cm.gray)  

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
