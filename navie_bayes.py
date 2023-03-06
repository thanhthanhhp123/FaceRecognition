import numpy as np
import math
import random
import os, fnmatch
import cv2

def navie_bayes(X_train, y_train, X_test):
    # Calculate p(y)
    classes, classes_count = np.unique(y_train, return_counts = True)
    classes_prob = classes_count / len(y_train)

    # Calculate p(X|y)
    classes_mean = []
    classes_std = []
    for i in range(len(classes)):
        class_X = np.array(X_train[np.where(y_train == classes[i])])
        classes_mean.append(np.mean(class_X, axis=0))   # TypeError: unsupported operand type(s) for +: 'NoneType' and 'NoneType'
        classes_std.append(np.std(class_X, axis=0))

    # Suppose X has Gaussian distribution
    y_pred = []
    for x in X_test:
        probs = []
        for i in range(len(classes)):
            prob = 1
            for j in range(len(x)):
                prob *= (1 / (np.sqrt(2 * np.pi * classes_std[i][j] ** 2))) * np.exp(-(x[j] - classes_mean[i][j]) ** 2 / (2 * classes_std[i][j] ** 2))
            probs.append(prob)
        # Choose the max prob
        y_pred.append(classes[np.argmax(probs)])

    return y_pred

class datasets():
    def __init__(self):
        pass
    def load_datasets():
        X = []; Y = []
        for i in os.listdir('Datasets/khai'):
            img = cv2.imread(os.path.join('Datasets/khai', i))
            X.append(img)
            Y.append(1)
        for i in os.listdir('Datasets/quan'):
            img = cv2.imread(os.path.join('Datasets/quan', i))
            X.append(img)
            Y.append(0)
        X = np.array(X)
        Y = np.array(Y)

        return X, Y

X, Y = datasets.load_datasets()

print("X = ", X)
print('Y.shape = ', Y.shape)
ratio = 0.7

X_train = X[:int(X.shape[0] * ratio)]
X_test = X[int(X.shape[0] * ratio):]

y_train = Y[:int(Y.shape[0] * ratio)]
y_test = Y[int(Y.shape[0] * ratio):]

y_pred = navie_bayes(X_train, y_train, X_test)
print(y_pred)
print(y_test)
