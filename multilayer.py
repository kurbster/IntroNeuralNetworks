#!/usr/bin/env python3
import numpy as np

sigmoid = lambda x: 1/(1+np.exp(-x))
linear_activation = lambda arr: arr >= 1
sigmoid_der = lambda y: y*(1 - y)

def forward_propagate(x, weights, func):
    '''
    for the number of layers we compute the dot
    product of the inputs and weights. Then pass
    that to the activation function. That returns
    the outpur for that layer.
    '''
    arr = [x]
    for i, w in enumerate(weights):
        sum = arr[i].dot(w)
        arr.append(func(sum))
    return arr 

def output_error(pred, y):
    err = error(pred, y)
    der_out = sigmoid_der(pred)
    return err * der_out

def error(pred, y, avg=False):
    return np.mean(abs(y-pred)) if avg else y-pred

def back_propagate(inputs, weights, delta_out, alpha):
    '''
    This function works by taking delta of the next layer
    and passing it to the previous one for the update. Hence
    backward progagating the error through the network.

    We start the loop at the first hidden layer right before the
    output and stop at index 0 which is the first input layer
    '''
    for i in range(len(inputs)-2, -1, -1):
        layerT = inputs[i].T
        # delta_out is the error of the output of this
        # layer. This is what will update the weights
        # for this layer
        layer_delta = layerT.dot(delta_out)
        # We backwards propagate the delta of this layer to the next
        # By taking the dot product of the error with the weights
        # times the derivative of the inputs. This is applying gradient descent
        delta_out = delta_out.dot(weights[i].T) * sigmoid_der(inputs[i])
        weights[i] += layer_delta*alpha

def train(x, y, weights, update=int(1e3), alpha=.1, max_itr=int(1e5), linear=False, plot=False):
    '''
    This function trains by forward propagating the inputs. Which are
    our predictions for each layer. Then backwards propagating the error
    and updating the weights each iteration. Then the loop predicts again
    with the new weights and updates again.

        forward_propagate ==> run the NN with weights
        back_propagate    ==> update the weights
    '''
    func = sigmoid if not linear else linear_activation
    errors = []
    for itr in range(max_itr):
        # get the inputs for each layer with forward
        # propagation. Pred is a list of predictions
        # for each layer ==> pred[-1] is output
        pred = forward_propagate(x, weights, func)
        delta_out = output_error(pred[-1], y)
        # update the weights with backward propagation
        # using the inputs == pred, weights, and
        # delta_out which is the error of our run
        back_propagate(pred, weights, delta_out, alpha)
        temp = error(pred[-1], y, avg=1)
        errors.append(temp)
        if itr % update == 0:
            print(f'Iteration {itr} error = {temp}')
    print(f'Iteration {max_itr} error = {errors[-1]}')
    if plot:
        import matplotlib.pyplot as plt
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.plot(range(max_itr), errors)
        plt.show()

def predict(instance, weights):
    return forward_propagate(instance, weights, sigmoid)[-1]
