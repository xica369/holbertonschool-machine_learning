#!/usr/bin/env python3

"""Likelihood"""

import numpy as np


def likelihood(x, n, P):
    """
    Find the probability that a patient who takes this drug will develop
    severe side effects.

    x follows a binomial distribution.
    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical probabilities
    of developing severe side effects

    Returns: a 1D numpy.ndarray with the likelihood of obtaining the data,
    x and n, for each probability in P, respectively
    """

    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, float)) or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.all(P > 1) and np.all(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    """l(P|x, n) = (n x) * P^x * (1 - P)^(n - x)"""

    comb = np.math.factorial(n)/(np.math.factorial(x) * np.math.factorial(n-x))
    likelihood = comb * pow(P, x) * pow(1 - P, n - x)

    return likelihood
