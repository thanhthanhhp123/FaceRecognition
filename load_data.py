import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

datas_folders = os.listdir('Datasets')
data_folders = []
for i in datas_folders:
    a = os.path.join('Datasets/', i)
    data_folders.append(a)

# print(data_folders)

for i in data_folders:
    a = os.listdir(i)
    print(np.array(a).shape)