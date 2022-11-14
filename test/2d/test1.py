import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PointEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PointEncoder, self).__init__()

        self.l1 = nn.Linear(input_size, input_size)
        self.l2 = nn.Linear(input_size, int(input_size//(1.2)))
        self.l3 = nn.Linear(int(input_size//(1.2)), int(input_size//(1.5)))
        self.l4 = nn.Linear(int(input_size//(1.5)), int(input_size//(1.7)))
        self.l5 = nn.Linear(int(input_size//(1.7)), hidden_size)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.l3(x))
        x = F.leaky_relu(self.l4(x))
        x = F.tanh(self.l5(x))
        return x
     

class EnvelopeNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EnvelopeNet, self).__init__()
        self.encoder = PointEncoder(input_size, hidden_size)
        self.l1 = nn.Linear(hidden_size+3, int(hidden_size//1.2))
        self.l2 = nn.Linear(int(hidden_size//1.2), int(hidden_size//1.5))
        self.l3 = nn.Linear(int(hidden_size//1.5), int(hidden_size//1.7))
        self.l4 = nn.Linear(int(hidden_size//1.7), int(hidden_size//2))
        self.l5 = nn.Linear(int(hidden_size//2), int(hidden_size//2.2))
        self.l6 = nn.Linear(int(hidden_size//2.2), 2)

    def forward(self, point_cloud, y, sigma):
        encpc = self.encoder(point_cloud)
        x = torch.cat([
            y, 
            torch.broadcast_to(torch.tensor([sigma], dtype=torch.float).to(device), (y.shape[0],1)),
            torch.broadcast_to(encpc, (y.shape[0], encpc.shape[1]))], dim=1)
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.l3(x))
        x = F.leaky_relu(self.l4(x))
        x = F.leaky_relu(self.l5(x))
        x = self.l6(x)
        return x

class CurveDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        for _, _, filenames in os.walk(root_dir):
            self.filenames = filenames

    def set_sigma(self, sigma=2):
        self.sigma = np.sqrt(sigma)
        
    def __getitem__(self, index):
        img = cv2.imread(f'{self.root_dir}/{self.filenames[index]}')
        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.random.randint(0,4):
            img2gray = cv2.rotate(img2gray, np.random.choice([
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
                cv2.ROTATE_180
            ]))
        std_img = img2gray//255

        content = np.argwhere(std_img==0)
        content_len = content.shape[0]

        point_cloud = content[np.random.randint(0, content_len, (256,))]

        sample_size = 16
        samples = point_cloud[np.random.randint(0, 256, (sample_size,))]
        samples = samples + (np.random.choice([-1,1], size=(sample_size,2)) * np.random.normal(loc=self.sigma, scale=self.sigma/3, size=(sample_size,2)))
        
        # sample_img = np.ones((64,64))

        # for i in range(content_len):
        #     sample_img[tuple(content[i])] = 0.9
        
        # for i in range(64):
        #     sample_img[tuple(point_cloud[i])] = 0.33
        
        # for i in range(sample_size):
        #     x = samples.astype(np.int32)
        #     sample_img[tuple(x[i])] = 0

        # plt.imshow(sample_img)
        # plt.show()    

        return (torch.tensor(samples, dtype=torch.float),
               torch.tensor(point_cloud, dtype=torch.float).flatten(),
               torch.tensor(content, dtype=torch.float))

    def __len__(self):
        return len(self.filenames)
        
        
def envelope_loss(output, samples, content, sigma=2, threshold=1, flag=False):
    '''
    point_cloud, output of shape (N, 2)
    '''
    n_outputs = len(output)
    n_points = len(content)
    broadcasted_content = content.broadcast_to((n_outputs, *content.shape))
    broadcasted_outputs = torch.repeat_interleave(output.reshape(output.shape[0], 1, output.shape[1]), torch.tensor([n_points]).to(device), axis=1)
    loss = ((broadcasted_content - broadcasted_outputs)/sigma)**2
    loss = torch.abs(torch.exp(-0.5 * loss.sum(axis=2)).sum(axis=1) - threshold).sum()
    return loss * torch.max(torch.norm(outputs-samples, dim=1), 1/torch.norm(outputs-samples, dim=1)).sum()

# train_dataset = CurveDataset('data/train')
# train_dataset.set_sigma(2)
# print(train_dataset[50])
# exit()

# hyperparameters
input_size = 256*2
hidden_size = 256
num_epochs = 10000
batch_size = 1
learning_rate = 0.00001
sigma = 2

try:
    model = torch.load('model.pth')
    model.eval()
except:
    model = EnvelopeNet(input_size, hidden_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# dataset
train_dataset = CurveDataset('./data/train')
train_dataset.set_sigma(sigma)
test_dataset = CurveDataset('./data/test')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

loss_history = []
envelope_loss_history = []
rolling_loss = 0
rolling_envelope_loss = 0

flag=False
# training loop
for epoch in range(num_epochs):
    for i, (samples, point_cloud, content) in enumerate(train_loader):
        samples = samples.reshape((-1,2)).to(device)
        point_cloud = point_cloud.to(device)
        content = content.reshape((-1,2)).to(device)

        outputs = model(point_cloud, samples, sigma)
        # loss = envelope_loss(outputs, samples, content, sigma=sigma)
        # # if flag:
        # #     loss = envelope_loss(outputs, samples, content, sigma=sigma)
        # #     rolling_envelope_loss = 0.95*rolling_envelope_loss + 0.05*loss
        # #     rolling_loss = 0.975*rolling_loss + 0.025*torch.norm(outputs-samples, dim=1).sum().item()
        # #     if rolling_loss>200:
        # #         flag=False
        # # else:
        # #     loss = torch.norm(outputs-samples, dim=1).sum()
        # #     rolling_loss = 0.975*rolling_loss + 0.025*loss.item()
        # #     if rolling_loss<30:
        # #         flag=True
        
        # rolling_loss = 0.975*rolling_loss + 0.025*loss.item()
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # loss_history.append(rolling_loss)
        # envelope_loss_history.append(rolling_envelope_loss)

        # if (i+1)%1 == 0:
        #     print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{len(train_dataset)}, loss = {rolling_loss:.4f} envelope_loss = {rolling_envelope_loss:.4f}')
    if epoch % 50 == 0:
        # torch.save(model, 'model.pth')
        envelope_loss(outputs, samples, content, sigma=sigma, flag=True)
        sample_img = np.ones((64,64))
        content_len = content.shape[0]

        samples = samples.cpu().numpy()
        content = content.cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        
        for j in range(content_len):
            x = content.astype(np.int32)
            sample_img[tuple(x[j])] = 0.9
            
        x = samples.astype(np.int32)
        print(x)
        for j in range(16):
            sample_img[tuple(x[j])] = 0.5
        
        x = outputs.astype(np.int32)
        print(x)
        for j in range(16):
            if 0<=x[j][0]<64 and 0<=x[j][1]<64:
                sample_img[tuple(x[j])] = 0

        plt.imshow(sample_img, cmap=plt.cm.gray)
        plt.show()    

        # loss_history = loss_history[-5000:]
        # plt.plot(loss_history)
        # plt.show()

        # envelope_loss_history = envelope_loss_history[-1000:]
        # plt.plot(envelope_loss_history)
        # plt.show()
