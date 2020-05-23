#!/usr/bin/env python3

"""
Function that performs the expectation maximization for a GMM:

X is a numpy.ndarray of shape (n, d) containing the data set
k is a positive integer containing the number of clusters
iterations is a positive integer containing the maximum number of iterations
for the algorithm
tol is a non-negative float containing tolerance of the log likelihood, used to
determine early stopping i.e. if the difference is less than or equal to tol
you should stop the algorithm
verbose is a boolean that determines if you should print information about the
algorithm
  If True, print Log Likelihood after {i} iterations: {l} every 10 iterations
  and after the last iteration
   {i} is the number of iterations of the EM algorithm
   {l} is the log likelihood

Returns: pi, m, S, g, l, or None, None, None, None, None on failure
pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
m is a numpy.ndarray of shape (k, d) containing the centroid means for each
cluster
S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for
each cluster
g is a numpy.ndarray of shape (k, n) containing the probabilities for each data
point in each cluster
l is the log likelihood of the model
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """EM: function that performs the expectation maximization for a GMM"""

    try:
        if X.ndim != 2:
            return None, None, None, None, None

        if X.shape[0] < 1 or X.shape[1] < 1:
            return None, None, None, None, None

        if not isinstance(k, int) or k < 1:
            return None, None, None, None, None

        if not isinstance(iterations, int) or iterations < 1:
            return None, None, None, None, None

        if not isinstance(tol, float) or tol < 0:
            return None, None, None, None, None

        if not isinstance(verbose, bool):
            return None, None, None, None, None

        pi, m, S = initialize(X, k)

        if pi is None or m is None or S is None:
            return None, None, None, None, None

        temp = 0
        for iter in range(iterations):

            g, lk = expectation(X, pi, m, S)
            pi, m, S = maximization(X, g)

            if verbose:
                if (iter % 10 == 0 or iter == iterations-1 or
                   abs(lk - temp) <= tol):
                    message = "Log Likelihood after {} iterations: {}"
                    print(message.format(iter, lk))

            if abs(lk - temp) <= tol:
                break

            temp = lk

    except Exception:
        return None, None, None, None, None

    return pi, m, S, g, lk
