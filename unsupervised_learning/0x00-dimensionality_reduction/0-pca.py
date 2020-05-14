#!/usr/bin/env python3

"""function that performs PCA on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """
    function that performs PCA on a dataset:

    X: numpy.ndarray of shape (n, d) where:
      n is the number of data points
      d is the number of dimensions in each point
      all dimensions have a mean of 0 across all data points
    var: fraction of the variance that the PCA transformation should maintain
    Returns: the weights matrix, W,
    that maintains var fraction of X‘s original variance
    W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality
    of the transformed X
    """

    U, sigm, V = np.linalg.svd(X)
    aculum_var = np.cumsum(sigm) / np.sum(sigm)
    r = np.argwhere(aculum_var >= var)[0, 0]

    return V[:r + 1].T
