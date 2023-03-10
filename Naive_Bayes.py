import numpy as np
import math
import random
import os, fnmatch
import cv2
from load_data import datasets
import matplotlib.pyplot as plt

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
                probs[i] = np.sum(np.log(gaussian + 1e-9)) + np.log(classes_prob[i])

            # Choose the max prob
            self.y_pred.append(classes[np.argmax(probs)])

        return self

    def accuracy(self, y_test): 
        accuracy = np.sum(self.y_pred == y_test) / len(y_test)
        return accuracy

# Load datas
X1, y1 = datasets.load_datasets_rgb()
X2, y2 = datasets.load_datasets_grayscale()

sizes = ['1280x720', '128x72', '64x36']
accuracy_rgb = []
accuracy_gray = []

#------------------------------------------------------------------
# Original size 1280 x 720
X1_train, y1_train, X1_test, y1_test = datasets.split_train_test(X1, y1, 0.2, random_state = 12) 
X2_train, y2_train, X2_test, y2_test = datasets.split_train_test(X2, y2, 0.2, random_state = 12)

nb1 = Naive_Bayes(X1, y1)
nb1.naive_bayes(X1_train, y1_train, X1_test)
accuracy_rgb.append(nb1.accuracy(y1_test))

nb2 = Naive_Bayes(X2, y2)
nb2.naive_bayes(X2_train, y2_train, X2_test)
accuracy_gray.append(nb2.accuracy(y2_test))

#------------------------------------------------------------------
# Size 128 x 72
size = (128, 72)

resized_images1 = np.zeros((200, 72, 128, 3))
resized_images2 = np.zeros((200, 72, 128))

for i in range(X1.shape[0]):
    resized_images1[i] = cv2.resize(X1[i], size, interpolation = cv2.INTER_AREA)

for i in range(X2.shape[0]):
    resized_images2[i] = cv2.resize(X2[i], size)

X1_train, y1_train, X1_test, y1_test = datasets.split_train_test(X1, y1, 0.2, random_state = 12) 
X2_train, y2_train, X2_test, y2_test = datasets.split_train_test(X2, y2, 0.2, random_state = 12)

nb1 = Naive_Bayes(X1, y1)
nb1.naive_bayes(X1_train, y1_train, X1_test)
accuracy_rgb.append(nb1.accuracy(y1_test))

nb2 = Naive_Bayes(X2, y2)
nb2.naive_bayes(X2_train, y2_train, X2_test)
accuracy_gray.append(nb2.accuracy(y2_test))

#------------------------------------------------------------------
#Size 64 x 36
size = (64, 36)

resized_images1 = np.zeros((200, 36, 64, 3))
resized_images2 = np.zeros((200, 36, 64))

for i in range(X1.shape[0]):
    resized_images1[i] = cv2.resize(X1[i], size, interpolation = cv2.INTER_AREA)

for i in range(X2.shape[0]):
    resized_images2[i] = cv2.resize(X2[i], size)

X1_train, y1_train, X1_test, y1_test = datasets.split_train_test(X1, y1, 0.2, random_state = 12) 
X2_train, y2_train, X2_test, y2_test = datasets.split_train_test(X2, y2, 0.2, random_state = 12)

nb1 = Naive_Bayes(X1, y1)
nb1.naive_bayes(X1_train, y1_train, X1_test)
accuracy_rgb.append(nb1.accuracy(y1_test))

nb2 = Naive_Bayes(X2, y2)
nb2.naive_bayes(X2_train, y2_train, X2_test)
accuracy_gray.append(nb2.accuracy(y2_test))


# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Naive-Bayes accuracy')

ax1.plot(sizes, accuracy_rgb, 'b')
ax1.set_title('with RGB')
ax1.set(xlabel = 'sizes', ylabel = 'accuracy')
ax1.set_ylim(0.5, 1.2)

ax2.plot(sizes, accuracy_gray, 'C7')
ax2.set_title('with gray scale')
ax2.set(xlabel = 'sizes')
ax2.set_ylim(0.5, 1.2)

plt.show()