#!/usr/bin/env python3
from sklearn import datasets
from multilayer import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

iris = datasets.load_iris()
x = iris.data
y = iris.target

def single_out():
    x = iris.data[:100]
    y = iris.target[:100]
    y = y.reshape(-1,1)
    layer1 = np.random.random((4,5))
    layer2 = np.random.random((5,5))
    layer3 = np.random.random((5,1))
    weights = [layer1, layer2, layer3]
    train(x, y, weights, max_itr=3000)

def classification():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    network = MLPClassifier(max_iter=2000,
            verbose=True,
            tol=1e-5,
            activation='logistic',
            batch_size=32,
            hidden_layer_sizes=(4,4))
    network.fit(x_train, y_train)
    pred = network.predict(x_test)
    score = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    print(f'Score = {score}')
    print(f'Matrix =\n{cm}')
single_out()
classification()
