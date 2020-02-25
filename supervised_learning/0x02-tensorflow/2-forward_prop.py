#!/usr/bin/env python3

"""Forward Propagation"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """function that creates the forward propagation graph
    for the neural network:
    x is the placeholder for the input data
    layer_sizes: list containing the number of nodes in each layer
    activations: with the activation functions for each layer
    Returns: the prediction of the network in tensor form"""

    prev = x
    for iter in range(len(activations)):
        n = layer_sizes[iter]
        activation = activations[iter]
        layer = create_layer(prev, n, activation)
        prev = layer

    return layer
