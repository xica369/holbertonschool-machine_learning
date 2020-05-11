#!/usr/bin/env python3

"""
function that calculates a correlation matrix:

C is a numpy.ndarray of shape (d, d) containing a covariance matrix
d is the number of dimensions

Returns a numpy.ndarray of shape (d, d) containing the correlation matrix
"""

import numpy as np


def correlation(C):
    """Correlation"""

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]
    correlation = np.ndarray((d, d))

    for i in range(d):
        for j in range(d):
            correlation[i][j] = C[i][j] / np.sqrt(C[i][i] * C[j][j])

    return correlation
