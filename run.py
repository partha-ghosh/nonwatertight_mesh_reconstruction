import os

os.system(f'python gen_mask_psr2.py train && python gen_mask_psr2.py test')
os.system(f'python train2.py configs/learning_based/demo_outlier.yaml')