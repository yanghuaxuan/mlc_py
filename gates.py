import numpy as np
from mlcv2 import Network, Layer, mse, mse_prime, tanh, tanh_prime
from random import randint

def batchtize(inputs, labels, batch_size):
    if len(inputs) != len(labels):
        raise
    if len(inputs) % batch_size != 0:
        raise
        
    c = 0
    chosen_indexes = []
    total_x = []
    total_y = []
    batch_x = []
    batch_y = []
    while c < len(inputs):
        i = randint(0, len(inputs)-1)
        while i in chosen_indexes:
            i = randint(0, len(inputs)-1)
        chosen_indexes += [i]
        if c != 0 and c % batch_size == 0:
            total_x += batch_x
            total_y += batch_y
            batch_x = []
            batch_y = []
        #print(f"inserting input: {inputs[i]}")
        batch_x.append(inputs[i])
        batch_y.append(labels[i])
        c+= 1
        #print("ran!")

    total_x += batch_x
    total_y += batch_y
    
    return (total_x, total_y)



net = Network(mse, mse_prime)
activator = (tanh, tanh_prime)

net.add(Layer(2, 3, activator))
net.add(Layer(3, 1, activator))

#x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
#y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
#net.fit(x_train, y_train, epochs=1000, learn_rate=0.1)

# training data
x_train = [np.array([[0,0]]), np.array([[0,1]]), np.array([[1,0]]), np.array([[1,1]])]
y_train = [np.array([[0]]), np.array([[1]]), np.array([[1]]), np.array([[0]])]
#net.fit(x_train, y_train, epochs=1000, learn_rate=0.1)

epochs = 1000
batch_size = 1
for i in range(epochs):
    print(f"+ Epoch: {i}")
    x_b, y_b = batchtize(x_train, y_train, batch_size)
    sum_batch_loss = 0
    for i in range(len(x_b)):
        #print(f"x_b[i]: {x_b}")
        sum_batch_loss += net.fit([x_train[i]], [y_train[i]], learn_rate=0.1)
    print(f"    loss: {sum_batch_loss / batch_size}")
