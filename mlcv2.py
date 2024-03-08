import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size)
        self.activate, self.d_activate = activation
    def forward(self, x):
        self.inputs = x
        self.outputs = self.activate(x @ self.weights + self.bias)
        return self.outputs
    # Returns the delta for this layer
    def err(self, delta_p, w_p):
        #print(f"delta: {delta_p}")
        delta = (delta_p @ w_p.T) * self.d_activate(self.outputs)
        #delta = (w_p @ delta_p.T) * self.d_activate(self.outputs)
        return delta
    # Perform backpropagation, returning the weights and bias gradients for this layer
    def back(self, delta):
        #print(f"self.inputs: {self.inputs.T}")
        #print(f"delta input: {delta}")
        #print(f"delta: {delta}")
        #print(f"self.inputs.T: {self.inputs.T}")
        
        grad_b = delta
        grad_w =  self.inputs.T @ delta 

        return (grad_w, grad_b)

class Network:
    def __init__(self, loss, d_loss):
        self.layers = []
        self.loss = loss
        self.d_loss = d_loss
    def add(self, layer):
        self.layers.append(layer)
    def fit(self, inputs, labels, learn_rate):
        # TODO: rename all mentions of epoch here to batch
        epoch_loss = 0 
        grad_w = []
        grad_b = []
        for x in enumerate(inputs):
            output = x[1]
            # forward propagate
            for layer in self.layers:
                output = layer.forward(output)

            epoch_loss += self.loss(output, labels[x[0]])
            
            lg_w = [] # gradients are stored in reverse of layers
            lg_b = []
            # backpropagate
            #delta = self.d_loss(output, labels[x[0]]) *  self.layers[-1].d_activate(output)
            delta = self.d_loss(output, labels[x[0]]) * self.layers[-1].d_activate(output) 
            #print(f"initial delta: {delta}")
            #print(f"# of layers: {len(self.layers)}")
            for i in range(len(self.layers)-1, -1, -1):
                #print(f"======At layer {i}")
                gw, gb = self.layers[i].back(delta)
                lg_w.append(gw)
                lg_b.append(gb)
                if i-1 >= 0:
                    delta = self.layers[i-1].err(delta, layer.weights)
            grad_w.append(lg_w)
            grad_b.append(lg_b)

        # Sum and average
        #grad_w = np.add.reduce(grad_w) / len(inputs)
        #grad_b = np.add.reduce(grad_b) / len(inputs)
        #print(f"grad_b: {grad_b}")
        grad_b_avg = []
        for i in range(len(grad_b[0])):
            b0 = grad_b[0][i]
            for j in range(1, len(grad_b)):
                #print(f"+=+= {grad_b[j][i]}")
                b0 += grad_b[j][i]
            grad_b_avg.append(b0 / len(grad_b))

        #print(f"grad_w: {grad_w}")
        grad_w_avg = []
        for i in range(len(grad_w[0])):
            w0 = grad_w[0][i]
            for j in range(1, len(grad_w)):
                #print(f"+=+= {grad_w[j][i]}")
                w0 += grad_w[j][i]
            #print(f"number of grad_ws: {len(grad_w)}")
            grad_w_avg.append(w0 / len(grad_w))
        #print(f"grad_w_avg: {grad_w_avg}")
            
        
        #print(f"grad_w: {grad_w}")
        #print(f"grad_b: {grad_b}")
        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            #print(f"layer weights: {layer.weights}")
            #print(f"grad_w_avg: {grad_w_avg}")
            layer.weights -= learn_rate * grad_w_avg[-i+1]
            layer.bias -= learn_rate * grad_b_avg[-i+1]
        
        # print(f"batch loss: {epoch_loss / len(inputs)}")
        return epoch_loss / len(inputs)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
    
def d_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-x) / ((1 + np.exp(-x))**2)

def mse(dy, y):
    return np.mean(np.power(dy-y, 2));

def mse_prime(dy, y):
    return 2*(dy-y)/dy.size;

def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;

from random import randint

