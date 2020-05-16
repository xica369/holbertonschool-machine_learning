#!/usr/bin/env python3

"""Q affinities"""

import numpy as np


def Q_affinities(Y):
    """
    that calculates the Q affinities:

    Y: numpy.ndarray of shape (n, ndim) with the low dimensional transformation
    of X
      n is the number of points
      ndim is the new dimensional representation of X

    Returns: Q, num
    Q: numpy.ndarray of shape (n, n) containing the Q affinities
    num: numpy.ndarray of shape (n, n) with the numerator of the Q affinities
    """
