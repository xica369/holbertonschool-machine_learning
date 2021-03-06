#!/usr/bin/env python3

"""
Function that tests for the optimum number of clusters by variance:

X is a numpy.ndarray of shape (n, d) containing the data set
kmin is a positive integer containing the minimum number of clusters
to check for (inclusive)
kmax is a positive integer containing the maximum number of clusters
to check for (inclusive)
iterations is a positive integer containing the maximum number of
iterations for K-means

Returns: results, d_vars, or None, None on failure
results is a list containing the outputs of K-means for each cluster size
d_vars is a list containing the difference in variance from the smallest
cluster size for each cluster size
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Optimize k"""

    try:
        if kmax is None:
            kmax = X.shape[0]

        if not isinstance(kmin, int) or kmin < 1:
            return None, None

        if not isinstance(kmax, int) or kmax <= kmin:
            return None, None

        if X.shape[0] < kmin or X.shape[0] < kmax:
            return None, None

        if X.ndim != 2:
            return None, None

        results = []
        d_vars = []
        C, clss = kmeans(X, kmin, iterations)
        smallest_var = variance(X, C)

        for k in range(kmin, kmax + 1, 1):
            C, clss = kmeans(X, k, iterations)
            results.append((C, clss))

            var = variance(X, C)
            d_vars.append(smallest_var - var)

    except Exception:
        return None, None

    return results, d_vars
