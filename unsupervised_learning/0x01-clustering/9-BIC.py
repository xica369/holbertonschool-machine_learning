#!/usr/bin/env python3

"""
Function that finds the best number of clusters for a GMM using the Bayesian
Information Criterion:

X is a numpy.ndarray of shape (n, d) containing the data set
kmin is a positive integer containing the minimum number of clusters to check
for (inclusive)
kmax is a positive integer containing the maximum number of clusters to check
for (inclusive)
iterations is a positive integer containing the maximum number of iterations
for the EM algorithm
tol is a non-negative float containing the tolerance for the EM algorithm
verbose is a boolean that determines if the EM algorithm should print
information to the standard output

Returns: best_k, best_result, l, b, or None, None, None, None on failure
best_k is the best value for k based on its BIC
best_result is tuple containing pi, m, S
pi is a numpy.ndarray of shape (k,) containing the cluster priors for the best
number of clusters
m is a numpy.ndarray of shape (k, d) containing the centroid means for the best
number of clusters
S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for
the best number of clusters
l is a numpy.ndarray of shape (kmax - kmin + 1) containing the log likelihood
for each cluster size tested
b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value for
each cluster size tested
Use: BIC = p * ln(n) - 2 * l
p is the number of parameters required for the model
n is the number of data points used to create the model
l is the log likelihood of the model
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """finds the best number of clusters for a GMM using the Bayesian
    Information Criterion"""

    try:
        n, d = X.shape

        if X.ndim != 2:
            return None, None, None, None

        if n < 1 or d < 1:
            return None, None, None, None

        if not isinstance(kmin, int) or kmin < 0:
            return None, None, None, None

        if not isinstance(kmax, int) or kmax <= kmin:
            return None, None, None, None

        if not isinstance(iterations, int) or iterations < 1:
            return None, None, None, None

        if not isinstance(tol, float) or tol < 0:
            return None, None, None, None

        if not isinstance(verbose, bool):
            return None, None, None, None

        k_results = []
        results = []
        likehood_total = []
        bics = []

        for ki in range(kmin, kmax + 1, 1):
            pi, m, S, g, lk = expectation_maximization(X, ki, iterations, tol,
                                                       verbose)

            k_results.append(ki)
            results.append((pi, m, S))
            likehood_total.append(lk)
            bic = ((ki*d * (d+1) / 2) + (d*ki) + ki - 1) * np.log(n) - 2*lk
            bics.append(bic)

        b = np.asarray(bics)
        lk = np.asarray(likehood_total)
        best = np.argmin(b)
        best_k = k_results[best]
        best_result = results[best]

    except Exception:
        return None, None, None, None

    return best_k, best_result, lk, b
