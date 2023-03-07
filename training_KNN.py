from load_data import datasets
from algorithms import KNN
import numpy as np
import matplotlib.pyplot as plt
import cv2

X, y, classes = datasets().load_datasets()

#------------------------------------------------------------------
#Original size 1280 x 720
X_train, y_train, X_test, y_test = datasets().split_train_test(X, y, test_ratio=0.2, random_state=12)
model = KNN(k = 5)
model.fit(X_train, y_train, X_test)
model.accuracy(y_test)

#------------------------------------------------------------------
#Size 128 x 72
size = (128, 72)
resized_images = np.zeros((200, 72, 128, 3))
for i in range(X.shape[0]):
    resized_images[i] = cv2.resize(X[i], size)
X_train, y_train, X_test, y_test = datasets().split_train_test(resized_images, 
                                                               y, test_ratio=0.2, random_state=12)
model = KNN(k = 5)
model.fit(X_train, y_train, X_test)
model.accuracy(y_test)

#------------------------------------------------------------------
#Size 64 x 36

size = (64, 36)
resized_images = np.zeros((200, 36, 64, 3))
for i in range(X.shape[0]):
    resized_images[i] = cv2.resize(X[i], size)
X_train, y_train, X_test, y_test = datasets().split_train_test(resized_images, 
                                                               y, test_ratio=0.2, random_state=12)
model = KNN(k = 5)
model.fit(X_train, y_train, X_test)
model.accuracy(y_test)