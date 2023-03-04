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

print(X.shape)