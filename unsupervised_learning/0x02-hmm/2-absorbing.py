#!/usr/bin/env python3

""""
Function that determines if a markov chain is absorbing:

P: square 2D numpy.ndarray of shape (n, n) representing the transition matrix
P[i, j] is the probability of transitioning from state i to state j
n is the number of states in the markov chain

Returns: True if it is absorbing, or False on failure
"""

import numpy as np


def absorbing(P):
    """
    Absorbing Chains
    """

    a = 1
    if a == 1:
        n = P.shape[0]

        if P.shape != (n, n) or n < 1:
            return None

        if not np.isclose(np.sum(P, axis=1), 1).all():
            return None

        if np.all(P <= 0):
            return None

        diag = P.diagonal()

