import numpy as np
import math
import random
from scipy import stats

class LinearRegression(object):
    def __init__(self, epochs, lr):
        self.epochs = epochs
        self.lr = lr
    def predict(self, X, w, b):
        y_pred = np.dot(X, w) + b
        return y_pred
    def compute_cost(self, X, y, w, b):
        m = X.shape[0]
        loss = np.square(self.predict(X, w, b) - y)
        cost = 1/(2*m) * np.sum(loss)
        return cost
    def compute_gradient(self, X, y, w, b):
        m = X.shape[0]
        dw = 1/m * (X.T @ (self.predict(X, w, b)- y))
        db = 1/m * np.sum(self.predict(X, w, b) - y)
        return dw, db
    def fit(self, X, y):
        w = np.random.randn(X.shape[1],1)
        b = 0
        J_history = []
        for i in range(self.epochs):
            dw, db = self.compute_gradient(X, y, w, b)
            w = w - self.lr * dw
            b = b - self.lr * db
            if i<100000:   
                    J_history.append(self.compute_cost(X, y, w, b))

            if i% math.ceil(self.epochs / 10) == 0:
                    print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        return w, b, J_history
    
class LogisticRegression(object):
  def __init__(self, eta = 0.01, epochs = 50):
    self.eta = eta
    self.epochs = epochs
  def sigmoid(self, z):
    s = 1/(1+np.exp(-z))
    return s
  def propagate(self, X, y, w, b):
    m = X.shape[1]
    A = self.sigmoid(np.dot(w.T, X) + b)
    cost = (- 1 / m) * np.sum(y * np.log(A) + (1 - y) * (np.log(1 - A)))

    dw = 1/m * np.dot(X, (A -y).T)
    db = 1/m * np.sum(A - y)

    cost = np.squeeze(cost)

    return dw, db, cost
  def optimize(self, X, y, w, b):
    costs = []
    for i in range(self.epochs):
      dw, db, cost = self.propagate(X, y, w, b)

      w = w - self.eta * dw
      b = b - self.eta * db
      costs.append(cost)
      if i % 100 == 0:
        print('Cost after epoch %i: %f' %(i, cost))

    return w, b, dw, db, costs
  def predict(self, X, w, b):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    # w = w.reshape(X.shape[0], 1)
    A = self.sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
      Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    return Y_prediction
  def fit(self, X, y):
    w = np.zeros(shape = (X.shape[0], 1))
    b = 0 
    w, b, dw, db, costs = self.optimize(X, y, w, b)
    d = {"costs": costs,
         "w" : w, 
         "b" : b,
         "learning_rate" : self.eta,
         "num_iterations": self.epochs}
    return d
  
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


class KNN(object):
  def __init__(self, k, p):
    self.k = k
    self.p = p
  
  def euclidean(self, v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))
  
  def fitData(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train

  def get_neighbors(self, test_row):
    dis = list()
    for (train_row, train_class) in zip(self.X_train, self.y_train):
      dist = self.euclidean(train_row, test_row)
      dis.append(dist, train_class)
    
    dis.sort(key=lambda x: x[0])

    neighbours = list()

    for i in range(self.k):
       neighbours.append(dis[i])
    
    return neighbours
  
  def predict(self, X_test):
     preds = []
     for test_row in X_test:
        nearest_neighbours = self.get_neighbours(test_row)
        majority = stats.mode(nearest_neighbours)[0][0]
        preds.append(majority)
        return np.array(preds)