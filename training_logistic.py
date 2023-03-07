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

#---------------------------------------------------
# #Original size 720 x 1280
X_train, y_train, X_test, y_test = datasets().split_train_test(X, Y, test_ratio=0.2, random_state=12)
X_train = X_train.reshape((X_train.shape[0], -1)).T / 255.
X_test = X_test.reshape((X_test.shape[0], -1)).T / 255.
model = LogisticRegression(epochs = 100, alpha=0.001)
his = model.fit(X_train, y_train, X_test, y_test, print_cost=True)
plt.title('Costs of 128 x 72')
plt.xlabel('Costs')
plt.ylabel('Epochs')
print(model.costs)
'''Train accuracy: 49.375
   Test accuracy: 52.5'''


#--------------------------------------------------------
#Size 128 x 72
size = (128, 72)
resized_images = np.zeros((200, 72, 128))
for i in range(X.shape[0]):
    resized_images[i] = cv2.resize(X[i], size)
    
X_train, y_train, X_test, y_test = datasets().split_train_test(resized_images, Y, test_ratio=0.2, random_state=12)

X_train = X_train.reshape((X_train.shape[0], -1)).T / 255.
X_test = X_test.reshape((X_test.shape[0], -1)).T / 255.
model = LogisticRegression(epochs = 200, alpha=0.0001)
his = model.fit(X_train, y_train, X_test, y_test, print_cost=True)
plt.title('Costs of 128 x 72')
plt.xlabel('Costs')
plt.ylabel('Epochs')
plt.plot(model.costs)

'''Train accuracy: 50.625
   Test accuracy: 47.5'''
   
#----------------------------------------------------------
#Size 64 x 36

size = (64, 36)
resized_images = np.zeros((200, 36, 64))
for i in range(X.shape[0]):
    resized_images[i] = cv2.resize(X[i], size)

X_train, y_train, X_test, y_test = datasets().split_train_test(resized_images, Y, test_ratio=0.2, random_state=12)

X_train = X_train.reshape((X_train.shape[0], -1)).T / 255.
X_test = X_test.reshape((X_test.shape[0], -1)).T / 255.
model = LogisticRegression(epochs = 200, alpha=0.0001)
his = model.fit(X_train, y_train, X_test, y_test, print_cost=True)
plt.title('Costs of 64 x 36')
plt.xlabel('Costs')
plt.ylabel('Epochs')
plt.plot(model.costs)
'''Train accuracy: 50.625
   Test accuracy: 47.5'''