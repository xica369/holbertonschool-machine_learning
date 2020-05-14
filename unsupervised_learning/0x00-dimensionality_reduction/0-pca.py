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

    n, d = X.shape

    mean = np.mean(X, axis=0)
    X = X - mean
    cov = np.dot(X.T, X) / (n - 1)

    vals, vects = np.linalg.eig(cov)

    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vects = vects[:, idx]
    vects *= -1

    sum_vals = np.sum(vals)

    # retention of the information per eigen value
    var_retention = vals / sum_vals
    acum_variance = np.cumsum(var_retention)

    r = 0
    for acum_var in acum_variance:
        r += 1
        if acum_var > var:
            break

    return vects[:, :r]
