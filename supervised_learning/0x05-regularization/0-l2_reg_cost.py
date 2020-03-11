#!/usr/bin/env python3

"""L2 Regularization Cost"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calculates the cost of a neural network with L2 regularization:

    cost is the cost of the network without L2 regularization
    lambtha is the regularization parameter
    weights is a dictionary of the weights and biases (numpy.ndarrays)
    L is the number of layers in the neural network
    m is the number of data points used
    Returns: the cost of the network accounting for L2 regularization"""

    for layer in range(L):
        sum_w = (np.linalg.norm(weights['W' + str(layer + 1)])) ** 2

    L2_cost = cost + lambtha / (2 * m) * sum_w

    return L2_cost
