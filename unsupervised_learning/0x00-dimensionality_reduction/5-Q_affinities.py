#!/usr/bin/env python3

"""Q affinities"""

import numpy as np


def Q_affinities(Y):
    """
    that calculates the Q affinities:

    Y: numpy.ndarray of shape (n, ndim) with the low dimensional transformation
    of X
      n is the number of points
      ndim is the new dimensional representation of X

    Returns: Q, num
    Q: numpy.ndarray of shape (n, n) containing the Q affinities
    num: numpy.ndarray of shape (n, n) with the numerator of the Q affinities
    """

    n = Y.shape[0]
    Q = np.zeros((n, n))

    sum_Y = np.sum(np.square(Y), axis=1, keepdims=True)

    # similar to (a-b)^2 = a^2 + b^2 - 2*a*b
    num = sum_Y + sum_Y.T - 2 * np.dot(Y, Y.T)
    num = (1 + num) ** -1

    # put diagonal in zeros
    num[range(n), range(n)] = 0

    Q = num / np.sum(num)

    return(Q, num)
