#!/usr/bin/env python3
import numpy as np
from multilayer import *

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0],[1],[1],[0]])
layer1 = np.array([[-.424,-.74,-.961],[.358,-.577,-.469]])
layer2 = np.array([[.1,.1,.1,.1,.1],[.2,-.1,-.15,.3,.05],[.1,.1,.1,.1,.1]])
layer3 = np.array([[.2,-.25,.3],[.2,-.25,.3],[.2,-.25,.3],[.2,-.25,.3],[.2,-.25,.3]])
layer4 = np.array([[-.017],[-.893],[.148]])
weights = [layer1, layer2, layer3, layer4]
train(x, y, weights, alpha=.1, activate=.5, max_itr=100)
