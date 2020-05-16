#!/usr/bin/env python3

"""Gradients"""

import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    that calculates the gradients of Y:

    Y: numpy.ndarray of shape (n, ndim) with the low dimensional transformation
    of X
    P: is a numpy.ndarray of shape (n, n) containing the P affinities of X

    Returns: (dY, Q)
    dY is a numpy.ndarray of shape (n, n) containing the gradients of Y
    Q is a numpy.ndarray of shape (n, n) containing the Q affinities of Y
    """

    n, m = Y.shape

    Q, num = Q_affinities(Y)
    dY = np.zeros((n, m))
    PQ = P - Q

    for iter in range(n):
        dY[iter, :] = np.sum(np.tile(PQ[:, iter] * num[:, iter],
                                     (m, 1)).T * (Y[iter, :]-Y), axis=0)

    return dY, Q
