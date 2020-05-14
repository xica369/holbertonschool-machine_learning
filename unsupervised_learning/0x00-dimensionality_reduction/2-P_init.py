#!/usr/bin/env python3

"""Initialize t-SNE"""

import numpy as np


def P_init(X, perplexity):
    """
    initializes all variables required to calculate the P affinities in t-SNE:

    X: numpy.ndarray of shape (n,d) with the dataset to be transformed by t-SNE
      n is the number of data points
      d is the number of dimensions in each point
    perplexity is the perplexity that all Gaussian distributions should have

    Returns: (D, P, betas, H)
    D: numpy.ndarray of shape (n, n) that calculates the pairwise distance
    between two data points
    P: a numpy.ndarray of shape (n, n) initialized to all 0‘s that
    will contain the P affinities
    betas: a numpy.ndarray of shape (n, 1) initialized to all 1’s that
    will contain all of the beta values
    H is the Shannon entropy for perplexity perplexity
    """

    n = X.shape[0]

    sum_x = np.sum(np.square(X), 1, keepdims=True)
    D = (np.add(np.add(-2 * np.dot(X, X.T), sum_x).T, sum_x))
    P = np.zeros((n, n), dtype="float64")
    betas = np.ones((n, 1), dtype="float64")
    H = np.log2(perplexity)

    return (D, P, betas, H)
