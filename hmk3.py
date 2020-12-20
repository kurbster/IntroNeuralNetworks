#!/usr/bin/env python3
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets

wine = datasets.load_wine()
x = wine.data
y = wine.target

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
network = MLPClassifier(verbose=True,
        activation='logistic',
        max_iter=2000,
        tol=1e-5)
network.fit(x_train, y_train)
pred = network.predict(x_test)
score = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
print(f'Score = {score}')
print(f'Matrix =\n{cm}')
