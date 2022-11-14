import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('data/train/10.png')
img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
std_img = img2gray//255

content_indices = np.argwhere(std_img==0)
content_indices_len = content_indices.shape[0]

sample_img = np.ones((64,64), dtype=np.uint8)

for i in range(64):
    index = np.random.randint(0, content_indices_len)
    sample_img[tuple(content_indices[index])] = 0

# plt.imshow(sample_img)
# plt.show()

s = ''
sigma = 3
for i in content_indices:
    s += f'e^{{-0.5\left(\left(\\frac{{\left(x-{i[0]}\\right)}}{{{sigma}}}\\right)^{{2}}+\left(\\frac{{\left(y-{i[1]}\\right)}}{{{sigma}}}\\right)^{{2}}\\right)}}+'

with open('env.txt', 'w') as f:
    f.write(s)