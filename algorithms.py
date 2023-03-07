import numpy as np
import math
import random
from scipy import stats
import copy
import collections
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
  def __init__(self, epochs, alpha):
    self.epochs = epochs
    self.alpha = alpha
  def sigmoid(self, z):
    s = 1/(1+np.exp(-z))
    return s
  def initialize_with_zeros(self, dim):
    w = np.zeros(shape = (dim, 1))
    b = 0
    return w, b
  def propagate(self, w, b, X, y):
    m = X.shape[1]
    #FORWARD PROPAGATION
    A = self.sigmoid(np.dot(w.T, X) + b)
    loss = y*np.log(A) + (1-y)*np.log(1-A)
    cost = -1/m * np.sum(loss, axis = 1, keepdims = True)

    #BACKWARD PROPAGATION
    dw = 1/m * np.dot(X, (A- y).T)
    db = 1/m * np.sum(A - y)

    cost = np.squeeze(np.array(cost))

    grads = {'dw': dw,
              'db': db}
    return grads, cost

  def optimize(self, w, b, X, y, print_cost = True):
      w = copy.deepcopy(w)
      b = copy.deepcopy(b)
      
      self.costs = []

      for i in range(self.epochs):
        grads, cost = self.propagate(w, b, X, y)
        dw = grads['dw']
        db = grads['db']

        #Updating
        w = w - self.alpha * dw
        b = b - self.alpha * db
        self.costs.append(cost)
        if i % math.ceil(self.epochs / 10) == 0:
              # Print the cost every 100 training iterations
              if print_cost:
                  print ("Cost after iteration %i: %f" %(i, cost))
      
      params = {"w": w,
                "b": b}
      
      grads = {"dw": dw,
              "db": db}
      
      return params, grads
  def predict(self, w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = self.sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
      Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    return Y_prediction
  def fit(self, X_train, y_train, X_test, y_test, print_cost = False):
    w, b = self.initialize_with_zeros(X_train.shape[0])
    parameters, grads = self.optimize(w, b, X_train, y_train, print_cost)
    w = parameters['w']
    b = parameters['b']

    Y_prediction_test = self.predict(w, b, X_test)
    Y_prediction_train = self.predict(w, b, X_train)

    if print_cost:
      print("Train accuracy: {}".format(100 - np.mean(np.abs(Y_prediction_train - y_train)) * 100))
      print('Test accuracy: {}'.format(100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))
    d = {
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : self.alpha,
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

class KNN():
  def __init__(self, k):
      self.k = k
  def euclidean_distance(self, x1, x2):
     return np.sqrt(np.sum((x1 - x2) ** 2))
  def accuracy(self, y_test):
    acc1 = np.sum(self.y_pred == y_test) / len(y_test) * 100
    print('Accuracy: {}'.format(acc1))
  def fit(self, X_train, y_train, X_test):
    self.y_pred = []
    for i in range(len(X_test)):
      distances = [self.euclidean_distance(X_test[i], x) for x in X_train]
      k_idx = np.argsort(distances)[:self.k]
      k_labels = [y_train[idx] for idx in k_idx]  
      most_common = collections.Counter(k_labels).most_common(1)
      self.y_pred.append(most_common[0][0])
    return np.array(self.y_pred)