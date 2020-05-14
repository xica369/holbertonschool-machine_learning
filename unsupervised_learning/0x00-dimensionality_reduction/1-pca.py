#!/usr/bin/env python3

"""PCA v2"""

import numpy as np


def pca(X, ndim):
    """
    that performs PCA on a dataset:

    X is a numpy.ndarray of shape (n, d) where:
      n is the number of data points
      d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X

    Returns:
    T numpy.ndarray of shape (n,ndim) with the transformed version of X
    """

    mean = np.mean(X, axis=0)
    X = X - mean

    U, sigm, V = np.linalg.svd(X)
    W = V[:ndim].T

    T = np.dot(X, W)

    return T
