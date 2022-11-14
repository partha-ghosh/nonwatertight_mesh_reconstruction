import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import mcubes
from model import UNET


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('model.pth')
model.eval()
model = model.to(device)

root_dir = '3d_data/x'
for _, _, f in os.walk(f'{root_dir}'):
    filenames = f

cross_sections_pc = []
with torch.no_grad():
    for i in range(256):
        skeleton = cv2.imread(f'{root_dir}/{i}.png')
        skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
        skeleton = skeleton//255
        cross_sections_pc.append(np.array(skeleton)[None,:])
    cross_sections_pc = np.array(cross_sections_pc)

    cross_sections_pc = torch.from_numpy(cross_sections_pc).type(torch.float).to(device)
    
    cross_sections = torch.zeros((256,1,256,256))
    for i in range(0,256,8):
        cross_sections[i:i+8] = model(cross_sections_pc[i:i+8]).cpu()
    
volume_data = np.zeros((256,256,256))
for i in range(len(cross_sections)-1):
    volume_data[i] = np.maximum(cross_sections[i][0], cross_sections[i+1][0])


vertices, triangles = mcubes.marching_cubes(volume_data, 0.5)

mcubes.export_mesh(vertices, triangles, "mesh.dae", "MyMesh")