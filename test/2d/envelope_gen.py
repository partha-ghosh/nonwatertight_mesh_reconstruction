import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

root = 'data/train' 
for _, _, f in os.walk(f'{root}/skeleton'):
    filenames = f

grid = np.array([(i,j) for i in range(0,128) for j in range(0,128)])
kernel = np.ones((5,5), np.uint8)

for filename in filenames:
    img = cv2.imread(f'{root}/skeleton/{filename}')
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, bwimg) = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    
    #bwimg = 255-bwimg
    envelop = cv2.dilate(bwimg,kernel,iterations = 1)
    

    #plt.imsave(f'{root}/skeleton/{filename}', bwimg, cmap=plt.cm.gray)
    plt.imsave(f'{root}/envelope/{filename}', envelop, cmap=plt.cm.gray)

# sample_img = np.ones((64,64))

# for i in range(content_len):
#     sample_img[tuple(content[i])] = 0.9

# for i in range(64):
#     sample_img[tuple(point_cloud[i])] = 0.33

# for i in range(sample_size):
#     x = samples.astype(np.int32)
#     sample_img[tuple(x[i])] = 0

# plt.imshow(std_img, cmap=plt.cm.gray)
# plt.show()    

# plt.imshow(envelope, cmap=plt.cm.gray)
# plt.show()    