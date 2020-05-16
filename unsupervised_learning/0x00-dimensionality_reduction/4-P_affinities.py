#!/usr/bin/env python3

"""P affinities"""

import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


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

    n = X.shape[0]

    # initialize D, P, beta and H
    D, P, beta, H = P_init(X, perplexity)

    # Iterate the data points
    for iter in range(n):

        # Calculate the Gaussian kernel and entropy for the current precision
        beta_min, beta_max = None, None
        Di = D[iter, np.concatenate((np.r_[0:iter], np.r_[iter+1:n]))]
        Hi, Pi = HP(Di, beta[iter])

        Hdiff = Hi - H

        # Evaluate whether the perplexity is within tolerance
        while abs(Hdiff) > tol:

            # Increase or decrease precision
            if Hdiff > 0:
                beta_min = beta[iter, 0]
                if beta_max is None:
                    beta[iter] = beta[iter] * 2
                else:
                    beta[iter] = (beta[iter] + beta_max) / 2
            else:
                beta_max = beta[iter, 0]
                if beta_min is None:
                    beta[iter] = beta[iter] / 2
                else:
                    beta[iter] = (beta[iter] + beta_min) / 2

            # again calculate Shannon and P affinities of the points entropy
            Hi, Pi = HP(Di, beta[iter])
            Hdiff = Hi - H

        # Set the iter row of P
        P[iter, np.concatenate((np.r_[0:iter], np.r_[iter+1:n]))] = Pi

    P = (P + P.T) / (2 * n)

    return P
