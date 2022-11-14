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
from pytorch3d.loss import chamfer_distance
import trimesh
from src.model import Encode2Points
from src.utils import load_model_manual

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        for _, _, filenames in os.walk(f'{root_dir}/pointclouds'):
            self.filenames = filenames
        
    def __getitem__(self, index):
        pcd = np.load(f'{self.root_dir}/pointclouds/{self.filenames[index]}')
        pcd_nwt = np.load(f'{self.root_dir}/pointclouds_nonwatertight/{self.filenames[index]}') 
        pcd_nwt = np.concatenate((pcd_nwt, [[0.0,0,0]]))
        return \
            torch.tensor(pcd, dtype=torch.float),\
            torch.tensor(pcd_nwt, dtype=torch.float)

    def __len__(self):
        return len(self.filenames)

# train_dataset = CustomDataset('data/train')
# print(train_dataset[0][0].shape)
# exit()

class Classifier(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()

        output_size = 256
        self.fc1 = nn.Linear(input_size, output_size)

        input_size = output_size
        output_size = int(input_size//1.5)
        self.fc2 = nn.Linear(input_size, output_size)

        input_size = output_size
        output_size = int(input_size//1.5)
        self.fc3 = nn.Linear(input_size, output_size)

        input_size = output_size
        output_size = int(input_size//1.5)
        self.fc4 = nn.Linear(input_size, output_size)

        input_size = output_size
        output_size = 1
        self.fc5 = nn.Linear(input_size, output_size)
        self.fc5.weight.data.fill_(-1)
        self.fc5.bias.data.fill_(0)
        self.fc4.weight.data.fill_(0.0)
        self.fc4.bias.data.fill_(0.01)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(torch.clip(self.fc5(x), -3, 3))
        return x

# hyperparameters
num_epochs = 10000
batch_size = 1
learning_rate = 5e-4

model = Classifier(32**4+3).to(device)
encoder = Encode2Points().to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()), lr=learning_rate)
criterion = chamfer_distance
# criterion = balanced_cross_entropy_loss
# criterion = tversky_loss
# criterion = dice_loss

state_dict = torch.load('enc.pt')
load_model_manual(state_dict['state_dict'], encoder)
encoder.eval()

# try:
#     model.load_state_dict(torch.load('model.pth'))    
#     model.eval()
#     # print('model loaded')
# except:
#     pass

# dataset
train_dataset = CustomDataset('./data/train')
test_dataset = CustomDataset('./data/test')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)



test_loss_history = []
loss_history = []
rolling_loss = 0
old_loss = 9999


# training loop
for epoch in range(num_epochs):
    for i, (pcds, pcds_nwt) in enumerate(train_loader):
        pcds = pcds.to(device)
        pcds_nwt = pcds_nwt.to(device)

        enc = encoder(pcds_nwt)['grid'].reshape(1,-1)
        pcds = pcds.reshape(-1,3)
        for b in range(0,pcds.shape[0],256):
            input_pcds = pcds[b:b+256]
            input_enc = enc.repeat(input_pcds.shape[0], 1)
            input_pcdnenc = torch.cat((input_pcds, input_enc), dim=1)
            if b == 0:
                vertex_selection = model(input_pcdnenc)
            else:
                vertex_selection = torch.cat((vertex_selection, model(input_pcdnenc)))
        pcds = (pcds * vertex_selection)[None,:]
        # print(psrs.shape, masks.shape, outputs.shape)
        loss, _ = criterion(pcds_nwt, pcds)
        
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
            pcds = pcds.to(device)
            pcds_nwt = pcds_nwt.to(device)

            enc = encoder(pcds_nwt)['grid'].reshape(1,-1)
            pcds = pcds.reshape(-1,3)
            for b in range(0,pcds.shape[0],256):
                input_pcds = pcds[b:b+256]
                input_enc = enc.repeat(input_pcds.shape[0], 1)
                input_pcdnenc = torch.cat((input_pcds, input_enc), dim=1)
                if b == 0:
                    vertex_selection = model(input_pcdnenc)
                else:
                    vertex_selection = torch.cat((vertex_selection, model(input_pcdnenc)))
            print(vertex_selection)
            pcds = pcds * vertex_selection
            # test_loss = criterion(pcds_nwt, psr)
            # test_loss_history.append(test_loss)

            trimesh.points.PointCloud(pcds.cpu()).export('predicted.obj')
            trimesh.points.PointCloud(pcds_nwt.reshape(-1,3).cpu()).export('original.obj')

        # loss_history = loss_history[-5000:] 
        # plt.plot(loss_history)
        # plt.savefig('loss.png')
        # plt.show()
                
        plt.figure()
        plt.plot(loss_history)
        plt.savefig('train_loss.png')
                    
        # plt.figure()
        # plt.plot(test_loss_history)
        # plt.savefig('test_loss.png')
