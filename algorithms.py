import numpy as np
import math
import random
import collections
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



# class KNN(object):
#   def __init__(self, k, p):
#     self.k = k
#     self.p = p
  
#   def euclidean(self, v1, v2):
#     return np.sqrt(np.sum((v1 - v2)**2))
  
#   def fitData(self, X_train, y_train):
#     self.X_train = X_train
#     self.y_train = y_train

#   def get_neighbors(self, test_row):
#     dis = list()
#     for (train_row, train_class) in zip(self.X_train, self.y_train):
#       dist = self.euclidean(train_row, test_row)
#       dis.append(dist, train_class)
    
#     dis.sort(key=lambda x: x[0])

#     neighbours = list()

#     for i in range(self.k):
#        neighbours.append(dis[i])
    
#     return neighbours
  
#   def predict(self, X_test):
#      preds = []
#      for test_row in X_test:
#         nearest_neighbours = self.get_neighbours(test_row)
#         majority = stats.mode(nearest_neighbours)[0][0]
#         preds.append(majority)
#         return np.array(preds)
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for i in range(len(X_test)):
        distances = [euclidean_distance(X_test[i], x) for x in X_train]
        k_idx = np.argsort(distances)[:k]
        k_labels = [y_train[idx] for idx in k_idx]  
        most_common = collections.Counter(k_labels).most_common(1)
        y_pred.append(most_common[0][0])
    return np.array(y_pred)
