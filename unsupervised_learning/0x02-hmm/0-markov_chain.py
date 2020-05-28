#!/usr/bin/env python3

"""
Function that determines the probability of a markov chain being in a
particular state after a specified number of iterations:

P: a square 2D numpy.ndarray of shape (n, n) representing the transition matrix
P[i, j]: the probability of transitioning from state i to state j
  n is the number of states in the markov chain
s:  numpy.ndarray of shape (1, n) representing the probability of starting
in each state
t: is the number of iterations that the markov chain has been through

Returns: a numpy.ndarray of shape (1, n) representing the probability of being
in a specific state after t iterations, or None on failure
"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Markov Chain
    """

    a = 1
    if a == 1:
        n = P.shape[0]
        if P.shape != (n, n):
            return None

        if s.shape != (1, n):
            return None

        if not isinstance(t, int) or t < 1:
            return None

        if not np.isclose(np.sum(P, axis=1), 1).all():
            return None

        if np.all(P <= 0):
            return None

        S = np.dot(s, np.linalg.matrix_power(P, t))

        return S
