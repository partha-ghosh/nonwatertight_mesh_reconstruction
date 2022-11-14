import os
import random
import numpy as np

train_test_split = 0.85
data_dir = '/home/scholar/tmp/shape_as_points/dat' 
mesh_dir = '/home/scholar/tmp/shape_as_points/data/Shapenet'
mesh_ext = '.off'

os.system('mkdir -p dat/train/meshes && rm -rf dat/train/meshes/*')
os.system('mkdir -p dat/test/meshes && rm -rf dat/test/meshes/*')

train_count = 0
test_count = 0

files = []

def find_files(root):
    for pwd, ds, fs in os.walk(root):
        files.extend([f'{pwd}/{fs[i]}' for i in range(len(fs)) if ('.off' in fs[i] and i<300)])
    for d in ds:
        find_files(f'{root}/{d}')

find_files(mesh_dir)
files = [files[i] for i in np.random.randint(0, len(files), 1000)]

for f in files:
    model = f
    if random.random() < train_test_split:
        print(f'cp {f} {data_dir}/train/meshes/{train_count}{mesh_ext}')
        os.system(f'cp {f} {data_dir}/train/meshes/{train_count}{mesh_ext}')
        train_count += 1
    else:
        print(f'cp {f} {data_dir}/test/meshes/{train_count}{mesh_ext}')
        os.system(f'cp {f} {data_dir}/test/meshes/{test_count}{mesh_ext}')
        test_count += 1