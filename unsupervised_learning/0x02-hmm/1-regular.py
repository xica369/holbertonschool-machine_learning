#!/usr/bin/env python3

"""
Determines the steady state probabilities of a regular markov chain:

P is a is a square 2D numpy.ndarray of shape (n, n) representing the
transition matrix
P[i, j] is the probability of transitioning from state i to state j
n is the number of states in the markov chain

Returns: a numpy.ndarray of shape (1, n) containing the steady state
probabilities, or None on failure
"""

import numpy as np


def regular(P):
    """
    Regular Chains
    """

    try:
        n = P.shape[0]

        if P.shape != (n, n):
            return None

        if not np.isclose(np.sum(P, axis=1), 1).all():
            return None

        if np.all(P <= 0):
            return None

        S_ini = np.zeros((1, n))
        S_ini[0][0] = 1
        S_prev = S_ini

        t = 0
        while True:
            t += 1

            P_pow = np.linalg.matrix_power(P, t)
            if np.any(P_pow <= 0):
                return None

            S = np.dot(S_ini, P_pow)
            if np.all(S == S_prev):
                break

            S_prev = S

        return S

    except Exception:
        return None
