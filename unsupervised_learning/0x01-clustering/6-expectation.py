#!/usr/bin/env python3

"""
Function that calculates the expectation step in the EM algorithm for a GMM:

X is a numpy.ndarray of shape (n, d) containing the data set
pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
m is a numpy.ndarray of shape (k, d) with the centroid means for each cluster
S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
for each cluster

Returns: g, lk, or None, None on failure
g is a numpy.ndarray of shape (k, n) containing the posterior probabilities for
each data point in each cluster
lk is the total log likelihood
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Expectation"""

    try:
        n, d = X.shape
        k = pi.shape[0]

        if X.ndim != 2:
            return None, None

        if n < 1 or d < 1 or k < 1 or k > n:
            return None, None

        if pi.shape != (k,):
            return None, None

        if m.shape != (k, d):
            return None, None

        if S.shape != (k, d, d):
            return None, None

        g = np.zeros((k, n))

        for ki in range(k):
            P = pdf(X, m[ki], S[ki])
            g[ki] = P * pi[ki]

        sum_g = np.sum(g, axis=0)
        log = np.log(sum_g)
        lk = np.sum(log)
        g /= sum_g

    except Exception:
        return None, None

    return g, lk
