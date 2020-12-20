#!/usr/bin/env python
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

boston = datasets.load_boston()
x = boston.data
y = boston.target
y = y.reshape(-1,1)
scalerX = MinMaxScaler()
x = scalerX.fit_transform(x)
scalerY = MinMaxScaler()
y = scalerY.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

network = MLPRegressor(verbose=False,
        max_iter=2000,
        hidden_layer_sizes=(100))
network.fit(x_train, y_train)
pred = network.predict(x_test)
abs_error = mean_absolute_error(y_test, pred)
sq_error = mean_squared_error(y_test, pred)
root_error = np.sqrt(sq_error)
print(f'abs error = {abs_error}\nMSE = {sq_error}\nRMSE = {root_error}')
test = x_test[0].reshape(1,-1)
pred = network.predict(test).reshape(-1,1)
print(scalerY.inverse_transform(pred))
print(scalerY.inverse_transform(y_test[0].reshape(-1,1)))
