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
