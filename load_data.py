
"""
Created on Sat Mar  4 09:54:25 2023

@author: thanhdeptrai
"""
import cv2
import numpy as np
import os 
from PIL import Image

np.random.seed(1)
datasets_folder = 'Datasets'


class datasets():
    def __init__(self):
        pass
    def load_datasets(self):
        X = []; Y = []
        for i in os.listdir('Datasets/quan'):
            img = Image.open(os.path.join('Datasets/quan', i)).convert('L')
            # img = cv2.imread(os.path.join('Datasets/quan', i))
            # img = img.resize((100, 100))
            X.append(np.array(img))
            Y.append(1)
        for i in os.listdir('Datasets/khai'):
            img = Image.open(os.path.join('Datasets/khai', i)).convert('L')
            # img = img.resize((100, 100))
            X.append(np.array(img))
            Y.append(0)
        X = np.array(X)
        Y = np.array(Y)
        classes = ['quan', 'khai']
        return X, Y, classes
    
    def split_train_test(self, X, y, test_ratio=0.2, random_state=None):
        n_test = int(len(X) * test_ratio)
        np.random.seed(random_state)
        test_indices = np.random.choice(len(X), size=n_test, replace=False)
        train_indices = np.array(list(set(range(len(X))) - set(test_indices)))
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        return X_train, y_train, X_test, y_test
