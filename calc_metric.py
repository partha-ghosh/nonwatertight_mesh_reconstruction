from cv2 import threshold
import pandas as pd
import numpy as np

df = pd.read_csv('metrick9.csv', header=None)

thresholds = ['0.0','12.7','25.4','38.','50.8','63.5','76.','88.','101.','114.','127.']

metric = dict()

metric['pred'] = list(df[[(True if 'pred' in x else False) for x in list(df[0])]].mean())
print(metric)
# for d in ['2D','3D']:
#     metric[d] = dict()
#     for th in thresholds:   
#         metric[d][th] = list(df[[(True if (d in y) and (th in y) else False) for y in list(df[0])]].mean())
