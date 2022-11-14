import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from model import UNET

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CurveDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        for _, _, filenames in os.walk(f'{root_dir}/skeleton'):
            self.filenames = filenames
        
    def __getitem__(self, index):
        skeleton = cv2.imread(f'{self.root_dir}/skeleton/{self.filenames[index]}')
        envelope = cv2.imread(f'{self.root_dir}/envelope/{self.filenames[index]}') 
        
        skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
        envelope = cv2.cvtColor(envelope, cv2.COLOR_BGR2GRAY)
        
        rotation_indicator = np.random.choice([
            -1, 
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180])
        if rotation_indicator!=-1:
            skeleton = cv2.rotate(skeleton, rotation_indicator)
            envelope = cv2.rotate(envelope, rotation_indicator)
        
        skeleton = skeleton//255
        envelope = envelope//255

        content = np.argwhere(skeleton==1)
        content_len = content.shape[0]

        sample_size = np.random.choice([256, 384, 512])
        point_cloud = content[np.random.randint(0, content_len, (sample_size,))]

        image_size = 128
        sample_skeleton = np.zeros((image_size,image_size))

        for i in range(sample_size):
            sample_skeleton[(
                max(min(int(point_cloud[i][0]+2*np.random.randn()), image_size-1), 0),
                max(min(int(point_cloud[i][1]+2*np.random.randn()), image_size-1), 0),
                )
                ] += 1
        
        for i in range(np.random.choice([10, 25, 50, 75, 100])):
            sample_skeleton[(np.random.randint(0,image_size), np.random.randint(0,image_size))] += 1

        sample_skeleton = (sample_skeleton-sample_skeleton.mean())/sample_skeleton.std()

        # plt.imshow(sample_skeleton)
        # plt.show()

        # plt.imshow(envelope)
        # plt.show()    

        return (
            torch.tensor(sample_skeleton, dtype=torch.float)[None,:],
            torch.tensor(envelope, dtype=torch.float)[None,:],
        )


    def __len__(self):
        return len(self.filenames)


# train_dataset = CurveDataset('data/train')
# print(train_dataset[10])
# exit()

# hyperparameters
num_epochs = 10000
batch_size = 8
learning_rate = 0.0001

# try:
#     model = torch.load('model.pth')
#     model.eval()
# except:
#     model = UNET(1, 1, features=[16, 32, 64, 128, 256, 512]).to(device)

model = UNET(1, 1, features=[16, 32, 64, 128, 256, 512]).to(device)
try:
    model.load_state_dict(torch.load('model.pkl')['model_state'])
    model.eval()
except:
  pass

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.L1Loss()

# dataset
train_dataset = CurveDataset('./data/train')
test_dataset = CurveDataset('./data/test')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

loss_history = []
rolling_loss = 0

# training loop
for epoch in range(num_epochs):
    for i, (samples, envelopes) in enumerate(train_loader):
        samples = samples.to(device)
        envelopes = envelopes.to(device)

        outputs = model(samples)
        loss = criterion(outputs, envelopes)
        
        rolling_loss = 0.975*rolling_loss + 0.025*loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(rolling_loss)

        if (i+1)%1 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {rolling_loss:.4f}')
    
    if epoch % 20 == 0:
        checkpoint = {
          "model_state": model.state_dict(),
        }
        torch.save(checkpoint, "model.pkl")
        
        plt.imsave('s.png', samples[0][0].cpu(), cmap=plt.cm.gray)
        # plt.show() 

        plt.imsave('o.png',outputs[0][0].detach().cpu(), cmap=plt.cm.gray)
        # plt.show()    

        # loss_history = loss_history[-5000:]
        # plt.plot(loss_history)
        # plt.show()
                
            
