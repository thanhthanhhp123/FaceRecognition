import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from algorithms import LogisticRegression

# datas_folders = os.listdir('Datasets')
# data_folders = []
# for i in datas_folders:
#     a = os.path.join('Datasets/', i)
#     data_folders.append(a)

# print(data_folders)

hung_folder = 'Datasets/Test/'
hung_folder_ = [os.path.join(hung_folder, i) for i in os.listdir(hung_folder)]
hung_img = []
for i in hung_folder_:
    hung_img.append(cv2.imread(i))

fig, ax = plt.subplots(1, len(hung_img))
for i in range(len(hung_img)):
    ax[i].imshow(hung_img[i], cmap = 'gray')
    ax[i].axis('off')

# print(np.array(hung_img))