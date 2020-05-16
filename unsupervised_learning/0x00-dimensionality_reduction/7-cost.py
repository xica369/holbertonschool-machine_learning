#!/usr/bin/env python3

"""Function Cost"""

import numpy as np


def cost(P, Q):
    """
    Function that calculates the cost of the t-SNE transformation:

    P is a numpy.ndarray of shape (n, n) containing the P affinities
    Q is a numpy.ndarray of shape (n, n) containing the Q affinities

    Returns: C, the cost of the transformation
    """
