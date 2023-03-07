import numpy as np
import math
import random
import os, fnmatch
import cv2
from load_data import datasets

class Naive_Bayes():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def naive_bayes(self, X_train, y_train, X_test):
        # Calculate p(y)
        classes, classes_count = np.unique(y_train, return_counts = True)
        classes_prob = classes_count / len(y_train)

        # Calculate means and standard deviation of X
        classes_mean = []
        classes_std = []
        for i in range(len(classes)):
            class_X = np.array(X_train[np.where(y_train == classes[i])])
            classes_mean.append(np.mean(class_X, axis=0))
            classes_std.append(np.std(class_X, axis=0))

        classes_mean = np.array(classes_mean)
        classes_std = np.array(classes_std)

        # Suppose X has Gaussian distribution, calculate p(y|X)
        self.y_pred = []

        probs = []
        for x in X_test:
            probs = np.ones(len(classes), dtype = float)
            for i in range(len(classes)):
                # Calculate p(X|y)
                gaussian = (1 / (np.sqrt(2 * np.pi * classes_std[i] ** 2))) * np.exp(-(x - classes_mean[i]) ** 2 / (2 * classes_std[i] ** 2))
                
                # Calculate the hypothesis h(x)
                probs[i] = np.sum(np.log(gaussian + 0.001)) + np.log(classes_prob[i])

            # Choose the max prob
            self.y_pred.append(classes[np.argmax(probs)])

        return self

    def accuracy(self, y_test): 
        accuracy = np.sum(self.y_pred == y_test) / len(y_test)
        return accuracy

# Load datas
X, y = datasets.load_datasets()

# Provide datas into 30% test, 70% train
X_train, y_train, X_test, y_test = datasets.split_train_test(X, y, 0.3)     

# Call Naive_Bayes class
nb = Naive_Bayes(X, y)
nb.naive_bayes(X_train, y_train, X_test)

# Compare predicted labels and real labels:
print('Predicted labels:')
print(nb.y_pred)
print('\nReal labels:')
print(y_test)

print('\nAccuracy of Naive Bayes: ', nb.accuracy(y_test) * 100, '%')