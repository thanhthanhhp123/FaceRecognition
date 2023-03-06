# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 09:54:25 2023

@author: thanhdeptrai
"""

from load_data import datasets
from algorithms import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import cv2


X, Y, classes = datasets().load_datasets()
X_train, y_train, X_test, y_test = datasets().split_train_test(X, Y, test_ratio=0.2, random_state=12)

#---------------------------------------------------
# #Original size 720 x 1280
X_train = X_train.reshape((X_train.shape[0], -1)).T / 255.
X_test = X_test.reshape((X_test.shape[0], -1)).T / 255.
print(X_train.shape) (160, 720, 1280, 3)
model = LogisticRegression(epochs = 80, alpha=0.01)
his = model.fit(X_train, y_train, X_test, y_test, print_cost=True)



#--------------------------------------------------------
#Size 128 x 128
# for i in range(X.shape[0]):
#     X[i] = cv2.resize(X[i], (0, 0), fx=0.1, fy=0.1)
# plt.imshow(X[0])
# plt.show()