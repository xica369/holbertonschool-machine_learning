#!/usr/bin/env python3

"""
Function that performs K-means on a dataset:
X is a numpy.ndarray of shape (n, d) containing the dataset
  n is the number of data points
  d is the number of dimensions for each data point
k is a positive integer containing the number of clusters
iterations is a positive integer containing the maximum number of iterations
hat should be performed
Returns: C, clss, or None, None on failure
C is a numpy.ndarray of shape (k, d) with the centroid means for each cluster
clss is a numpy.ndarray of shape (n,) containing the index of the cluster in
C that each data point belongs to
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """K-means"""

    try:
        C = initialize(X, k)

        if not isinstance(iterations, int) or iterations < 1:
            return None, None

        d = X.shape[1]
        low = np.amin(X, axis=0)
        high = np.amax(X, axis=0)
        c_temp = C.copy()

        # calculate centroides
        for i in range(iterations):

            # calculate Euclidean distance between data and centroids
            distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))

            # take the shortest distance
            clss = np.argmin(distances, axis=0)

            # update position of centroids
            for k in range(C.shape[0]):

                # If a cluster does not have data points, reset it
                if (len(X[clss == k]) == 0):
                    C[k, :] = np.random.uniform(low, high, size=(1, d))
                else:
                    C[k, :] = (X[clss == k].mean(axis=0))

            # check if the centroids have stabilized
            if (c_temp == C).all():
                return C, clss

            c_temp = C.copy()

    except Exception:
        return None, None

    return C, clss


def initialize(X, k):
    """Initialize K-means"""

    try:
        if not isinstance(k, int) or k < 1:
            return None, None

        if X.ndim != 2:
            return None, None

        if X.shape[0] < 1 or X.shape[1] < 1:
            return None, None

        if k > X.shape[0]:
            return None, None

        d = X.shape[1]

        # minimum values of X along each dimension in d
        low = np.amin(X, 0)

        # maximum values of X along each dimension in d
        high = np.amax(X, 0)

        # initialize with a multivariate uniform distribution
        initialization = np.random.uniform(low=low, high=high, size=(k, d))

    except Exception:
        return None, None

    return initialization
