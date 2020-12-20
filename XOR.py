#!/usr/bin/env python3
import numpy as np
from multilayer import *

def XOR1():
    x = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0],[1],[1],[0]])
    layer1 = np.random.random((2,3))
    layer2 = np.random.random((3,1))
    weights = [layer1, layer2]
    train(x, y, weights, alpha=.3, update=int(1e4), plot=1)

def XOR2():
    '''
    This is an exact NN for the XOR operator and doesn't
    need any training
    '''
    x = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0],[1],[1],[0]])
    layer1 = np.array([[1,-1],[-1,1]])
    layer2 = np.array([[1],[1]])
    weights = [layer1, layer2]
    print('Exact predicion\n', forward_propagate(x, weights, linear_activation)[-1])

XOR1()
XOR2()
