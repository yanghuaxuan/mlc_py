# Python prototype of mlc

import random

import numpy as np
from math import exp
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
        self.biases = rng.random(size)
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
        
# Performs a feed forward, and calcualtes the gradients for the Cost function
def backprop(net: Network, x: [float], y: [float]) -> None:
    # Forward and store z and its activations
    z_list = []
    a_list = []
    for i in range(len(net.layers)):
        if i == 0:
            l = net.layers[i]
            z = l.weights @ x + l.biases
            z_list.append(z)
            a_list.append(l.activate(z))
        else:
            l = net.layers[i]
            z = l.weights @ a_list[i-1] + l.biases
            z_list.append(z)
            a_list.append(l.activate(z))

    print("debug print weights in each layers")
    for i in range(len(net.layers)):
        print(f"layer {i} weights: {net.layers[i].weights}")
    # initial condition
    delta = d_cost(a_list[-1], y) * d_sigmoid(z_list[-1])
    
    # iterate through layers in reverse
    for l in range(2, len(net.layers)):
        w = net.layers[-l+1].weights
        z = z_list[-l]
        print(f"w: {w}")
        print(f"w.transpose(): {w.transpose()}")
        print(f"delta: {delta}")
        # Err
        delta = np.dot(w.transpose(), delta) * d_sigmoid(z)

    # Error of l layer

def fit(net: Network) -> None:
    pass

def __main__():
    ## AND Gate Example
    # Create a Neural Network of input_shape 2
    net0 = Network(input_shape=2, layers=[3,2, 1])
    backprop(net0, [1,1], [1])
__main__()
