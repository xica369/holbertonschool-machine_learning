#!/usr/bin/env python3

"""
function that calculates the mean and covariance of a data set:

X is a numpy.ndarray of shape (n, d) containing the data set:
n is the number of data points
d is the number of dimensions in each data point

Returns: mean, cov:
  mean: numpy.ndarray of shape (1, d) containing the mean of the data set
  cov: numpy.ndarray of shape (d, d) with the covariance matrix of the data set
"""

import numpy as np


def mean_cov(X):
    """Mean and Covariance"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0)

    n = X.shape[0]
    x = X - mean

    cov = np.dot(x.T, x) / (n - 1)

    return mean, cov
