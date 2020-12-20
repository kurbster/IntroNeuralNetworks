#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv('data/diabetes.csv')
df = MinMaxScaler().fit_transform(df)
x = df[:,:-1]
y = df[:,-1]

convert = lambda a:torch.tensor(a, dtype=torch.float)
x_train, x_test, y_train, y_test = map(convert, train_test_split(x, y, test_size = 0.20))
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

dataset = TensorDataset(x_train, y_train)
t_loadr = DataLoader(dataset, batch_size=10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

network = nn.Sequential(
    nn.Linear(8,5),
    nn.Sigmoid(),
    nn.Linear(5,5),
    nn.Sigmoid(),
    nn.Linear(5,1),
    nn.Sigmoid())

network.to(device)
loss_func = nn.BCELoss()
opti = torch.optim.Adam(network.parameters(), lr=.001)
itrs = 2000
for itr in range(itrs):
    running_loss = 0
    for data in t_loadr:
        ins, outs = data[0].to(device), data[1].to(device)
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
    if itr%100 == 0:
        print(f'Epoch: {itr} loss: {running_loss/len(t_loadr)}')

print(f'Epoch: {itrs} loss: {running_loss/len(t_loadr)}')
network.eval()
pred = network.forward(x_test)
pred = np.array(pred >= 0.5)
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
