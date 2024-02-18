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
        self.biases = rng.random((size, 1))
        # self.biases = rng.random(size)
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
    a_list = [np.array([x]).T]
    grad_b = [np.zeros(l.biases.shape) for l in net.layers]
    grad_w = [np.zeros(l.weights.shape) for l in net.layers]

    print("debug print weights in each layers")
    for i in range(len(net.layers)):
        print(f"layer {i} weights: {net.layers[i].weights}")

    # Note: net.layers does not store the input layer, so a_list[i] is the "previous" activation layer
    for i in range(len(net.layers)):
        l = net.layers[i]
        print("=+=+=+=")
        print(f"At i : {i}")
        print(f"weights: {l.weights}")
        print(f"weights shape: {l.weights.shape}")
        print(f"a[i]: {a_list[i]}")
        print(f"activations shape: {a_list[i].shape}")
        print(f"w * a[i-1]: {l.weights @ a_list[i]}")
        print(f"l.biases: {l.biases}")
        print(f"l.biases.T: {l.biases.T}")
        z = l.weights @ a_list[i] + l.biases
        print(f"final z shape: {z.shape}")
        z_list.append(z)
        a_list.append(l.activate(z))

    # initial condition
    delta = d_cost(a_list[-1], y) * d_sigmoid(z_list[-1])
    print(f"delta initial: {delta}")
    grad_b[-1] = delta
    print(f"grad_b initial: {grad_b[-1]}")
    print(f"a_list[-2]: {a_list[-2]}")
    print(f"a_list[-2] type: {type(a_list[-2])}")
    print(f"a_list[-2] shape: {a_list[-2].shape}")
    print(f"a_list[-2]: {a_list[-2]}")
    print(f"a_list[-2].T: {a_list[-2].T}")
    grad_w[-1] = a_list[-2] @ delta
    print(f"grad_w initial: {grad_w[-1]}")
    
    # Since net.layers does not include the input layer, but the list of activations do
    print(f"total layers in network: {len(net.layers)}")
    for l in range(len(net.layers) - 2, -1, -1):
        print("=-=-=-=")
        print(f"At layer: {l}")
        print("=-=-=-=")
        w = net.layers[l+1].weights
        z = z_list[l]
#        print(f"w: {w}")
        print(f"w+1.T: {w.T}")
        print(f"delta before: {delta}")
        # Err
        #delta = (w.transpose() @ delta) * d_sigmoid(z)
        #delta = (w.transpose() @ delta) * d_sigmoid(z)
        delta = (w.T @ delta) * d_sigmoid(z)
        print(f"delta: {delta}")
        grad_b[l] = delta
        print(f"a_list: {a_list}")
        print(f"a_list[l]: {a_list[l]}")
        print(f"a_list[l] (transposed): {a_list[l].T}")
        grad_w[l] = a_list[l] @ delta.T

    # Error of l layer
    print()
    print()
    print(f"grad_w: {grad_w}")

def fit(net: Network) -> None:
    pass

def __main__():
    ## AND Gate Example
    # Create a Neural Network of input_shape 2
    net0 = Network(input_shape=2, layers=[3,2, 1])
    backprop(net0, [1,1], [1])
__main__()
