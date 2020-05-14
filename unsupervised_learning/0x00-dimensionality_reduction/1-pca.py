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

    n, d = X.shape

    # calcular matriz covarianza
    mean = np.mean(X, axis=0)
    X = X - mean
    C = np.dot(X.T, X) / (X.shape[0] - 1)

    # obtener valores y vectores propios
    vals, vects = np.linalg.eig(C)

    # ordena vals y vects en forma descendente
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vects = vects[:, idx]

    vects = -1 * vects[:, :ndim]
    T = np.dot(X, vects)

    return T.astype("float64")
