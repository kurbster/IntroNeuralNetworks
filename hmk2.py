#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from multilayer import *

df = pd.read_csv('credit_data.csv')
df.dropna(inplace=True)
x = df.iloc[:, 1:4].values
y = df.iloc[:, -1].values

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = y.reshape(-1,1)
layer1 = np.random.random((3,10))
layer2 = np.random.random((10,5))
layer3 = np.random.random((5,1))
weights = [layer1, layer2, layer3]
train(x, y, weights, update=int(1e3), max_itr=int(1e4), plot=1, alpha=.01)
