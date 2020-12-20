#!/usr/bin/env python3
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from perceptron import *

scaler = MinMaxScaler()
x = np.array([[18,2],[20,3],[21,4],[35,15],[36,16],[38,18]])
y = np.array([0,0,0,1,1,1])
x = scaler.fit_transform(x)
weights = np.array([0.0,0.0])
train(x, y, weights, alpha=0.01)
print(weights)
test = np.array([[17,5],[25,8],[45,10],[31,20]])
test = scaler.transform(test)
print(predict(x, weights, 1))
print(predict(test, weights, 1))
