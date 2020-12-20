#!/usr/bin/env python3
import torch
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target

convert = lambda a:torch.tensor(a, dtype=torch.float)
x_train, x_test, y_train, y_test = map(convert, train_test_split(x, y, test_size = 0.25))
# then we must create a dataset and data loader
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
dataset = TensorDataset(x_train, y_train)
t_loadr = DataLoader(dataset, batch_size=10)

'''
our NN structure
inputs = 30 features
output = 1, tumor or not

    30->16->16->1
'''
network = nn.Sequential(nn.Linear(
    in_features=30, out_features=16),
    nn.Sigmoid(),
    nn.Linear(16,16),
    nn.Sigmoid(),
    nn.Linear(16,1),
    nn.Sigmoid())

loss_func = nn.BCELoss()
opti = torch.optim.Adam(network.parameters(), lr=.001)
itrs = 100
for itr in range(itrs):
    running_loss = 0
    for data in t_loadr:
        ins, outs = data
        # get the gradient for the batch
        opti.zero_grad()
        # forward propagate
        pred = network.forward(ins)
        # get the loss
        loss = loss_func(pred, outs)
        # backward propagate
        loss.backward()
        # update the weights
        opti.step()

        running_loss += loss.item()
    if itr%10 == 0:
        print(f'Epoch: {itr} loss: {running_loss/len(t_loadr)}')

print(f'Epoch: {itrs} loss: {running_loss/len(t_loadr)}')
network.eval()
pred = network.forward(x_test)
pred = np.array(pred > 0.5)
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
