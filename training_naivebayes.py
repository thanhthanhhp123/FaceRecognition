from load_data import datasets
from algorithms import Naive_Bayes
import numpy as np
import matplotlib.pyplot as plt
import cv2

X, y, classes = datasets().load_datasets()

#------------------------------------------------------------------
#Original size 1280 x 720
X_train, y_train, X_test, y_test = datasets().split_train_test(X, y, test_ratio=0.2, random_state=12)
nb = Naive_Bayes(X, y)
nb.naive_bayes(X_train, y_train, X_test)

# Compare predicted labels and real labels:
print('Predicted labels:')
print(nb.y_pred)
print('\nReal labels:')
print(y_test)

print('\nAccuracy of Naive Bayes: ', nb.accuracy(y_test) * 100, '%')

#------------------------------------------------------------------
#Size 128 x 72
size = (128, 72)
resized_images = np.zeros((200, 72, 128))
for i in range(X.shape[0]):
    resized_images[i] = cv2.resize(X[i], size)
X_train, y_train, X_test, y_test = datasets().split_train_test(resized_images, 
                                                               y, test_ratio=0.2, random_state=12)
nb = Naive_Bayes(X, y)
nb.naive_bayes(X_train, y_train, X_test)

# Compare predicted labels and real labels:
print('Predicted labels:')
print(nb.y_pred)
print('\nReal labels:')
print(y_test)

print('\nAccuracy of Naive Bayes: ', nb.accuracy(y_test) * 100, '%')
#------------------------------------------------------------------
#Size 64 x 36

size = (64, 36)
resized_images = np.zeros((200, 36, 64))
for i in range(X.shape[0]):
    resized_images[i] = cv2.resize(X[i], size)
X_train, y_train, X_test, y_test = datasets().split_train_test(resized_images, 
                                                               y, test_ratio=0.2, random_state=12)
nb = Naive_Bayes(X, y)
nb.naive_bayes(X_train, y_train, X_test)

# Compare predicted labels and real labels:
print('Predicted labels:')
print(nb.y_pred)
print('\nReal labels:')
print(y_test)

print('\nAccuracy of Naive Bayes: ', nb.accuracy(y_test) * 100, '%')