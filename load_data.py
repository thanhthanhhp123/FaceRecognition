import cv2
import numpy as np
import os

class datasets():
    def __init__(self):
        pass
    def load_datasets():
        X = []; y = []
        for i in os.listdir('Datasets/khai'):
            img = cv2.imread(os.path.join('Datasets/khai', i))
            X.append(img)
            y.append(1)
        for i in os.listdir('Datasets/quan'):
            img = cv2.imread(os.path.join('Datasets/quan', i))
            X.append(img)
            y.append(0)
        X = np.array(X)
        y = np.array(y)

        return X, y

    def split_train_test(X, y, test_ratio = 0.3, random_state = None):   
        n_test = int(len(X) * test_ratio)
        np.random.seed(random_state)

        test_indices = np.random.choice(len(X), size = n_test, replace = False)
        train_indices = np.array(list(set(range(len(X))) - set(test_indices)))

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        return X_train, y_train, X_test, y_test