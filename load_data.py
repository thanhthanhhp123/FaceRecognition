import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 

np.random.seed(1)
datasets_folder = 'Datasets'

# X = []
# Y = []
# hung_folder = os.listdir('Datasets/Hung')
# for i in hung_folder:
#     img = cv2.imread(os.path.join('Datasets/Hung/', i))
#     X.append(img)
#     Y.append(1)

# for i in os.listdir('Datasets/Truong'):
#     img = cv2.imread(os.path.join('Datasets/Truong/', i), 1)
#     X.append(img)
#     Y.append(0)

# X = np.array(X)
# Y = np.array(Y)

class datasets():
    def __init__(self):
        pass
    def load_datasets(self):
        X = []; Y = []
        for i in os.listdir('Datasets/Hung'):
            img = cv2.imread(os.path.join('Datasets/Hung', i))
            X.append(img)
            Y.append(1)
        for i in os.listdir('Datasets/Truong'):
            img = cv2.imread(os.path.join('Datasets/Truong', i))
            X.append(img)
            Y.append(0)
        X = np.array(X)
        Y = np.array(Y)
        classes = ['Hung', 'Truong']
        return X, Y, classes