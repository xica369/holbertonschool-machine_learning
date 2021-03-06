#!/usr/bin/env python3

"""
Function that calculates the maximization step in the EM algorithm for a GMM:

X is a numpy.ndarray of shape (n, d) containing the data set
g is a numpy.ndarray of shape (k, n) containing the posterior probabilities
for each data point in each cluster

Returns: pi, m, S, or None, None, None on failure
pi is a numpy.ndarray of shape (k,) with the updated priors for each cluster
m is a numpy.ndarray of shape (k, d) with the updated centroid means for
each cluster
S is a numpy.ndarray of shape (k, d, d) containing the updated covariance
matrices for each cluster
"""

import numpy as np


def maximization(X, g):
    """Calculate Maximization"""

    try:
        n, d = X.shape
        k = g.shape[0]

        if X.ndim != 2 or g.ndim != 2:
            return None, None, None

        if n < 1 or d < 1 or k < 1 or n < k:
            return None, None, None

        if g.shape[1] != n:
            return None, None, None

        if not np.isclose(np.sum(g, axis=0), 1).all():
            return None, None, None

        pi = np.zeros((k,))
        m = np.zeros((k, d))
        S = np.zeros((k, d, d))

        for ki in range(k):
            Nk = np.sum(g[ki])
            pi[ki] = Nk / n
            m[ki] = np.matmul(g[ki].reshape(1, n), X).sum(axis=0) / Nk
            S[ki] = np.matmul(g[ki] * (X - m[ki]).T, (X - m[ki])) / Nk

    except Exception:
        return None, None, None

    return pi, m, S
