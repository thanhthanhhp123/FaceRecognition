# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 09:54:25 2023

@author: admin
"""

from load_data import datasets
from algorithms import LogisticRegression
import numpy as np
import matplotlib.pyplot

X, Y, classes = datasets().load_datasets()
m_train = X.shape[0]
X_flatten = X.reshape((m_train, -1)).T
# print(X_flatten.shape)

X_flatten = X_flatten/255.
