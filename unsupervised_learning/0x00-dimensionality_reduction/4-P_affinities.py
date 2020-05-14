#!/usr/bin/env python3

"""P affinities"""

import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP
global betas


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    that calculates the symmetric P affinities of a data set:

    X: numpy.ndarray of shape (n,d) with the dataset to be transformed by t-SNE
      n is the number of data points
      d is the number of dimensions in each point
    perplexity: perplexity that all Gaussian distributions should have
    tol: maximum tolerance allowed (inclusive) for the difference in Shannon
    entropy from perplexity for all Gaussian distributions

    Returns:
    P, a numpy.ndarray of shape (n, n) containing the symmetric P affinities
    """

    D, P, betas, H = P_init(X, perplexity)

    n = betas.shape[0]

    for iter in range(n):
        x = D[iter]
        prob = search_beta(x, iter, perplexity, tol, D)
        P[iter] = prob
    return P


def search_beta(x, iter, perplexity, tol, D):
    """ search beta with binary search"""

    beta_min, beta_max = 0, np.inf
    Hi, Pi[iter, 1:] = HP(D[iter, 1:], betas[iter])
    perp = 2 ** Hi
    perp_diff = perplexity - per
    times = 0
    hit_upper_limit = False

    while(abs(perp_diff) > tol) and (times < 50):
        if perp_diff > 0:
            if hit_upper_limit:
                beta_min = betas[iter]
                betas[iter] = (beta_min + beta_max) / 2
            else:
                beta_min, beta_max = betas[iter], betas[iter] * 2
                betas[iter] = beta_max
        else:
            beta_max = betas[iter]
            betas[iter] = (beta_min + beta_max) / 2
            hit_upper_minit = True
        Hi, Pi[iter, 1:] = HP(D[iter, 1:], betas[iter])
        perp = 2 * Hi
        perp_diff = perplexity - perp
        times = times + 1

    return Pi
