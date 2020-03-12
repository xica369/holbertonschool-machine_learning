#!/usr/bin/env python3

"""Forward Propagation with Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """hat conducts forward propagation using Dropout:

    X is a np.ndarray of shape (nx, m) with the input data for the network
    nx is the number of input features
    m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    L the number of layers in the network
    keep_prob is the probability that a node will be kept
    Returns: a dictionary with the outputs of each layer and
    the dropout mask used on each layer"""

    cache = {}
    cache['A0'] = X
    z = np.dot(weights['W1'], X) + weights['b1']

    for iter in range(1, L):
        A = np.tanh(z)
        D = np.random.binomial(n=1, p=keep_prob, size=z.shape)
        A = A * D / keep_prob
        z = np.dot(weights['W'+str(iter+1)], A) + weights['b'+str(iter+1)]

        cache['A'+str(iter)] = A
        cache['D'+str(iter)] = D

    A = np.exp(z)/(np.sum(np.exp(z), axis=0, keepdims=True))
    cache['A'+str(L)] = A

    return cache
