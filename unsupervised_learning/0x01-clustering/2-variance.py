#!/usr/bin/env python3

"""
Function that calculates the total intra-cluster variance for a data set:

X is a numpy.ndarray of shape (n, d) containing the data set
C is a numpy.ndarray of shape (k, d) containing the centroid means
for each cluster

Returns: var, or None on failure
var is the total variance

"""

import numpy as np


def variance(X, C):
    """Variance"""

    try:
        if X.shape[1] != C.shape[1]:
            return None

        if C.shape[0] > X.shape[0] or C.shape[0] < 1:
            return None

        if X.ndim != 2 or C.ndim != 2:
            return None

        if X.shape[1] < 1 or C.shape[1] < 1:
            return None

        distances = np.sqrt(np.sum(pow((X - C[:, np.newaxis]), 2), -1))
        min_distance = np.min(distances, axis=0)
        var = pow(min_distance, 2).sum()

    except Exception:
        return None

    return var
