#!/usr/bin/env python3
import numpy as np
activation_func = lambda a, tol: a >= tol 
def calc(inputs, weights):
    return inputs.dot(weights)

def predict(x, weights, activate):
    return activation_func(calc(x, weights),activate)

def error(pred, y, avg=False):
    return np.mean(abs(y-pred)) if avg else y-pred

def train(x, y, weights, tol=0, activate=1, alpha=.1, max_itr=1e6):
    itr = 0
    pred = predict(x, weights, activate)
    err = error(pred, y)
    while (sum(abs(err)) > tol):
        for i in range(len(x)):
            act = y[i]
            py = pred[i]
            if (act != py):
                weights += (alpha*x[i]*err[i])
                pred = predict(x, weights, activate)
                err = error(pred, y)
        itr += 1
        if (itr >= max_itr):
            break;
    print(f'After {itr} generations\nweights = {weights}\nerror = {err}')

if __name__ == '__main__':
    inputs = np.array([[35,85,45,15],[35,25,45,15]])
    weights = np.array([-.8,.4,.5,-.6])
    y = np.array([0,1])
    train(inputs, y, weights, alpha=.01)
