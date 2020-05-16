#!/usr/bin/env python3

"""Function t-SNE"""

import numpy as np


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    that performs a t-SNE transformation:

    X: numpy.ndarray of shape (n,d) with the dataset to be transformed by t-SNE
      n is the number of data points
      d is the number of dimensions in each point
    ndims is the new dimensional representation of X
    idims is the intermediate dimensional representation of X after PCA
    perplexity is the perplexity
    iterations is the number of iterations
    lr is the learning rate

    Returns:
    Y, a numpy.ndarray of shape (n, ndim)
    containing the optimized low dimensional transformation of X
    """
