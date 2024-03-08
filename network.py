# Python prototype of mlc

import random

import numpy as np
from collections.abc import Callable

rng = np.random.default_rng(seed=32)

'''
Activation functions
'''
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
def d_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-x) / ((1 + np.exp(-x))**2)

'''
Cost functions
'''
def cost(a: np.ndarray, y: np.ndarray):
    return (a - y) ** 2
# d_cost/da
def d_cost(a: np.ndarray, y: np.ndarray):
    return 2 * (a - y)

'''
Neural net layer class
'''
class Layer:
    def __init__(self, size: int, input_shape: int, activation: Callable[[float], float]):
        self.size = size # number of neurons in layer 
        self.input_shape = input_shape 
        self.weights = rng.random((size, input_shape))
        self.biases = rng.random((size, 1))
        self.activate = activation

'''
Contains the input "layer", and the hidden and output layers.
- The input layer is simply a vector
- The hidden and output layers are of class Layer
'''
class Network:
    def __init__(self, input_shape: int, layers: list[int]):
            self.inputs = np.empty(input_shape)
            self.layers = []
            for i in range(len(layers)):
                if i == 0:
                    self.layers.append(Layer(layers[i], input_shape, sigmoid))
                else:
                    self.layers.append(Layer(layers[i], layers[i-1], sigmoid))

'''
Perform feedforward and return the activations of the output layer
'''
def forward(net: Network, x: list[float]) -> np.ndarray:
    a = np.array([x]).T
    grad_b = [np.zeros(l.biases.shape) for l in net.layers]
    grad_w = [np.zeros(l.weights.shape) for l in net.layers]

    for i in range(len(net.layers)):
        l = net.layers[i]
        z = l.weights @ a + l.biases
        a = l.activate(z)

    return a
        
'''
Performs a feed forward, and calculates the gradients for the Cost function
Returns a tuple of weight and bias gradients
'''
def backprop(net: Network, x: list[float], y: list[float]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    # Forward and store z and its activations
    z_list = []
    a_list = [np.array([x]).T]
    grad_b = [np.zeros(l.biases.shape) for l in net.layers]
    grad_w = [np.zeros(l.weights.shape) for l in net.layers]

    for i in range(len(net.layers)):
        l = net.layers[i]
        # Note: net.layers does not store the input layer, so a_list[i] is the "previous" activation layer
        z = l.weights @ a_list[i] + l.biases
        z_list.append(z)
        a_list.append(l.activate(z))

    # initial condition
    delta = d_cost(a_list[-1], y) * d_sigmoid(z_list[-1])
    grad_b[-1] = delta
    grad_w[-1] = (a_list[-2] @ delta).T
    for l in range(len(net.layers) - 2, -1, -1):
        w = net.layers[l+1].weights
        z = z_list[l]
        delta = (w.T @ delta) * d_sigmoid(z)
        grad_b[l] = delta
        # net.layers does not include the input layer, but the list of activations do,  so a_list[l] is correct
        grad_w[l] = (a_list[l] @ delta.T).T

    return (grad_w, grad_b)

'''
Perform SGD
'''
def SGD(net: Network, training_data: list[tuple[list[float], list[float]]], mini_batch_size: int, epochs: int, learn_rate: float) -> None:
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        print("--------------")
        random.shuffle(training_data)
        c = 0
        grad_b = [np.zeros(l.biases.shape) for l in net.layers]
        grad_w = [np.zeros(l.weights.shape) for l in net.layers]
        avg_loss = 0
        for x, y in training_data:
            if (c != mini_batch_size):
#                avg_loss += 
                (back_w, back_b) = backprop(net, x, y)
                grad_w = [gw+bw for gw, bw in zip(grad_w, back_w)]
                grad_b = [gb+bb for gb, bb in zip(grad_b, back_b)]
                c += 1
            else:
                grad_w = [w * (learn_rate/mini_batch_size) for w in grad_w]
                grad_b = [b * (learn_rate/mini_batch_size) for b in grad_b]

                for i in range(len(net.layers)):
                    net.layers[i].weights -= grad_w[i]
                    net.layers[i].biases -= grad_b[i]
                grad_b = [np.zeros(l.biases.shape) for l in net.layers]
                grad_w = [np.zeros(l.weights.shape) for l in net.layers]
                c = 0

