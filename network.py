# Python prototype of mlc

import random

import numpy as np
from collections.abc import Callable

rng = np.random.default_rng(seed=32)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
def d_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-x) / ((1 + np.exp(-x))**2)

def cost(a: np.ndarray, y: np.ndarray):
    return (a - y) ** 2
def d_cost(a: np.ndarray, y: np.ndarray):
    return 2 * (a - y)

class Layer:
    def __init__(self, size: int, input_shape: int, activation: Callable[[float], float]):
        self.size = size # number of neurons in layer 
        self.input_shape = input_shape 
        self.weights = rng.random((size, input_shape))
        self.biases = rng.random((size, 1))
        self.activate = activation

# Creates a Network()
class Network:
    def __init__(self, input_shape: int, layers: [int]):
            self.inputs = np.empty(input_shape)
            self.layers = []
            for i in range(len(layers)):
                if i == 0:
                    self.layers.append(Layer(layers[i], input_shape, sigmoid))
                else:
                    self.layers.append(Layer(layers[i], layers[i-1], sigmoid))

# Feed inputs and iteratively activate all neurons in each layer
def forward(net: Network, inputs: [float]) -> np.ndarray:
    pass
        
'''
Performs a feed forward, and calculates the gradients for the Cost function
Returns a tuple of weight and bias gradients
'''
def backprop(net: Network, x: [float], y: [float]) -> tuple[np.ndarray, np.ndarray]:
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
#    print(f"grad_w[-1]: {grad_w[-1]}")
    for l in range(len(net.layers) - 2, -1, -1):
#        print(f"At layer {l}")
        w = net.layers[l+1].weights
        z = z_list[l]
        delta = (w.T @ delta) * d_sigmoid(z)
        grad_b[l] = delta
        # net.layers does not include the input layer, but the list of activations do,  so a_list[l] is correct
        grad_w[l] = (a_list[l] @ delta.T).T

#    print(f"final grad_w: {grad_w}")
    return (grad_w, grad_b)

def SGD(net: Network, training_data: [tuple[[float], [float]]], mini_batch_size: int, epochs: int, learn_rate: float) -> None:
    for _ in range(epochs):
        random.shuffle(training_data)
        c = 0
        grad_b = [np.zeros(l.biases.shape) for l in net.layers]
        grad_w = [np.zeros(l.weights.shape) for l in net.layers]
        for x, y in training_data:
            if (c != mini_batch_size):
                (back_w, back_b) = backprop(net, x, y)
#                print(f"grad_w: {grad_w}")
#                print(f"back_w: {back_w}")
                grad_w = [gw+bw for gw, bw in zip(grad_w, back_w)]
                grad_b = [gb+bb for gb, bb in zip(grad_b, back_b)]
                c += 1
            else:
#                print(f"grad_w: {grad_w}")
#                grad_w /= mini_batch_size
                grad_w = [w / mini_batch_size for w in grad_w]
#                grad_b /= mini_batch_size
                grad_b = [b / mini_batch_size for b in grad_b]
#                print(f"grad_w: {grad_w}")
#                print(f"back_w: {back_w}")

                for i in range(len(net.layers)):
#                    print(f"l.weights: {net.layers[i].weights}")
#                    print(f"grad_w[i]: {grad_w[i]}")
                    net.layers[i].weights -= grad_w[i]
                    net.layers[i].biases -= grad_b[i]
#                print(f"grad_w_batch: {grad_w_batch}")
#                net.weights -= grad_w_batch
#                net.bias -= grad_b_batch
                grad_b = [np.zeros(l.biases.shape) for l in net.layers]
                grad_w = [np.zeros(l.weights.shape) for l in net.layers]
                c = 0

def __main__():
    ## AND Gate Example
    # Create a Neural Network of input_shape 2
    net0 = Network(input_shape=2, layers=[3,2, 1])
#    print(backprop(net0, [1,1], [1]))
    SGD(net0, training_data=[([1,1], [1]), ([1,0], [0]), ([0,1], [0]), ([0,0],[0])], mini_batch_size=2, epochs=100, learn_rate=1e-3)
__main__()
