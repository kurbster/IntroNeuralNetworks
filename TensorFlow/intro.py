#!/usr/bin/env python3
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils import np_utils
from tensorflow.keras.datasets import mnist

def convert(sample):
    _s = sample.shape
    return sample.reshape(_s[0], _s[1] * _s[2]).astype('float32') / 255

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# reshape the the datasets to single array of
# pixels rather then X x Y pixels
# and normalize from 0 to 1
x_train = convert(x_train)
x_test = convert(x_test)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# After data preprocessing
'''
    Our NN is
    784 input pixels, 2 layers of 397 neurons, 10 outputs

        784->397->397->10
'''
network = Sequential()
network.add(Dense(
    input_shape=(784,),
    units=397,
    activation='relu'))
network.add(Dense(
    units=397,
    activation='relu'))
network.add(Dense(
    units=10,
    activation='softmax'))
network.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
history = network.fit(x_train, y_train, batch_size=128, epochs=50)
accuracy = network.evaluate(x_test, y_test)
