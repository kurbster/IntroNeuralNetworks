#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.utils import np_utils
from tensorflow.keras.datasets import fashion_mnist

def convert(sample):
    _s = sample.shape
    return sample.reshape(_s[0], _s[1] * _s[2]).astype('float32') / 255.0

def convert3D(sample):
    _s = sample.shape
    return sample.reshape(-1, _s[1], _s[2], 1).astype('float32') / 255.0

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

def MLP():
    X_train = convert(x_train)
    X_test = convert(x_test)
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
    network.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    history = network.fit(X_train, y_train,
        batch_size=128, epochs=20,
        validation_data=(X_test, y_test))

def CNN():
    '''
    input:
        start with 28x28x1 image
    first convolution:
        5x5x32 convolution layer
    first pooling:
        2x2 pooling layer
    second convolution:
        5x5x64 convolution layer
    second pooling:
        2x2 pooling layer
    flatten layer:
        flatten layers for dense layer
    Dense layer 1:
    Dropout layer:
    Dense layer 2:
        This is the output layer.

          Conv1 -> Pool1 -> Conv2 -> Pool2 -> Flat -> NN -> Out
    Para  5x5x32 -> 2x2  -> 5x5x64 -> 2x2  -> ... -> 1024-> 10
    Input 28x28x1 -> 28x28x32 -> 14x14x32 -> 14x14x64 -> 7x7x64 -> 1024 -> 10
    '''
    X_test = convert3D(x_test)
    X_train = convert3D(x_train)
    network = Sequential()
    network.add(Conv2D(
        filters=32, kernel_size=(5,5),
        strides=(1,1), padding='same',
        data_format='channels_last',
        name='conv1', activation='relu'))
    network.add(MaxPool2D(pool_size=(2,2), name='pool1'))
    network.add(Conv2D(
        filters=64, kernel_size=(5,5),
        strides=(1,1), padding='same',
        name='conv2', activation='relu'))
    network.add(MaxPool2D(pool_size=(2,2), name='pool2'))
    network.add(Flatten())
    network.add(Dense(
        units=1024,
        activation='relu',
        name='fc1'))
    network.add(Dropout(rate=0.5))
    network.add(Dense(
        units=10,
        activation='softmax',
        name='out'))
    network.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    history = network.fit(X_train, y_train,
            batch_size=128, epochs=10,
            validation_data=(X_test, y_test))
MLP()
CNN()
