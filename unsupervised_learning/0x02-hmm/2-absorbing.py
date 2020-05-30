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

    try:
        n = P.shape[0]

        if P.shape != (n, n) or n < 1:
            return False

        if not np.isclose(np.sum(P, axis=1), 1).all():
            return False

        if np.all(P <= 0):
            return False

        diag = P.diagonal()

        if np.all(diag == 1):
            return True

        if np.all(diag != 1):
            return False

        # check if there are two group of nodes that are not connecting
        for pos in range(n):
            e = P[:pos + 1, pos + 1:]
            f = P[pos + 1:, :pos + 1]
            if pos == n - 1:
                e = P[n - 1, : n]
                f = P[: n, n - 1]
            if np.all(e == 0) and np.all(f == 0):
                return False

        # check if all absorbent nodes only connect with themselves
        cont1 = 0
        cont2 = 0
        for pos in range(n):
            col = P[:pos, pos]
            _col = P[pos + 1:, pos]
            index = P[pos][pos]
            if index == 1:
                cont1 += 1
                if np.all(col == 0) and np.all(_col == 0):
                    cont2 += 1

        if cont1 == cont2:
            return False

        return True

    except Exception:
        return True
