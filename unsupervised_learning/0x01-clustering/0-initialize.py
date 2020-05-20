#!/usr/bin/env python3

"""
Function that initializes cluster centroids for K-means:

X is a numpy.ndarray of shape (n, d) containing the dataset that
will be used for K-means clustering
n is the number of data points
d is the number of dimensions for each data point
k is a positive integer containing the number of clusters

Returns: a numpy.ndarray of shape (k, d) containing the initialized centroids
for each cluster, or None on failure
"""

import numpy as np


def initialize(X, k):
    """Initialize K-means"""

    if not isinstance(k, int) or k < 1:
        return None

    try:
        d = X.shape[1]

        # minimum values of X along each dimension in d
        low = np.amin(X, 0)

        # maximum values of X along each dimension in d
        high = np.amax(X, 0)

        # initialize with a multivariate uniform distribution
        initialization = np.random.uniform(low=low, high=high, size=(k, d))

    except Exception:
        return None

    return initialization
