#!/usr/bin/env python3

"""Batch Normalization"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a neural network
    using batch normalization:

    Z is a numpy.ndarray of shape (m, n) that should be normalized
    m is the number of data points
    n is the number of features in Z
    gamma: numpy.ndarray of shape (1, n)
    containing the scales used for batch normalization
    beta: numpy.ndarray of shape (1, n) containing the offsets
    used for batch normalization
    epsilon is a small number used to avoid division by zero
    Returns: the normalized Z matrix"""

    mean = Z.mean(0)
    varianza = Z.var(0)

    z_n = (Z - mean) / np.sqrt(np.power(varianza, 2) + epsilon)
    z = gamma * z_n + beta

    return z
