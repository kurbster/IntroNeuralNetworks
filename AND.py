#!/usr/bin/env python3
import numpy as np
from perceptron import *

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,0,0,1])
weights = np.array([0.0,0.0])
train(x, y, weights)
