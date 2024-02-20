from network import Network, SGD, forward

# AND Gate neural network
if __name__ == "__main__":
    ## AND Gate Example
    # Create a Neural Network of input_shape 2
    net0 = Network(input_shape=2, layers=[3,2, 1])
#    print(backprop(net0, [1,1], [1]))
    SGD(net0, training_data=[([1,1], [1]), ([1,0], [0]), ([0,1], [0]), ([0,0],[0])], mini_batch_size=3, epochs=10000, learn_rate=2)
    print(f"x: [1,1]: {forward(net0, [1,1])}")
    print(f"x: [1,0]: {forward(net0, [1,0])}")
    print(f"x: [0,1]: {forward(net0, [0,1])}")
    print(f"x: [0,0]: {forward(net0, [0,0])}")
