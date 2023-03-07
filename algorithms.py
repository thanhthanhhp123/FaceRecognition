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
  
class NaiveBayes(object):
  def __init__(self, data):
    self.data = data
  def encode(self):
    self.classes = []
    for i in range(len(self.data)):
      if self.data[i][-1] not in self.classes:
          self.classes.append(self.data[i][-1])
    for i in range(len(self.classes)):
        for j in range(len(self.data)):
          if self.data[j][-1] == self.classes[i]:
              self.data[j][-1] = i
    return self.data
   
   #Splitting data
  def split(self, ratio):
    train_num = int(len(self.data) * ratio)
    train = []
    test = list(self.data)
    while len(train) < train_num:
       idx = random.randrange(len(test))
       train.append(test.pop(idx))
    
    return train, test
  #Group the data rows under the each class yes or no in dict
  def groupUnderClass(self):
    dict = {}
    for i in range(len(self.data)):
      if(self.data[i][-1] not in dict):
        dict[self.data[i][-1]] = []
      dict[self.data[i][-1]].append(self.data[i])
    return dict  
  
  def mean(self, numbers): 
     return sum(numbers) / float(len(numbers))
  
  def std_dev(self, numbers):
     avg = self.mean(numbers)
     variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
     return math.sqrt(variance)
  
  def MeanAndStdDevForClass(self):
    self.info = {}
    dict = self.groupUnderClass()
    for classValue, instaces in dict.items():
      self.info[classValue] = self.MeanAndStdDevForClass(instaces)
  
  def calculateGaussianProbability(self, x, mean, stdev):
    expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * expo
  
  def calculateClassProbabilities(self, test):
    probabilities = {}
    for classValue, classSummaries in self.info.items():
      probabilities[classValue] = 1
      for i in range(len(classSummaries)):
          mean, std_dev = classSummaries[i]
          x = test[i]
          probabilities[classValue] *= self.calculateGaussianProbability(x, mean, std_dev)
    return probabilities
  
  def predict(self, test):
    probabilities = self.calculateClassProbabilities(test)
    bestLabel, bestProb = None, -1
    for classValue, prob in probabilities.items():
        if bestLabel is None or prob > bestProb:
           bestProb = prob
           bestLabel = classValue
    return bestLabel
  
  def getPredictions(self, test):
    predictions = []
    for i in range(len(test)):
       result = self.predict(self.info, test[i])
       predictions.append(result)
    return predictions



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