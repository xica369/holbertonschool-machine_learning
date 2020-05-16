#!/usr/bin/env python3

"""Gradients"""

import numpy as np


def grads(Y, P):
    """
    that calculates the gradients of Y:

    Y: numpy.ndarray of shape (n, ndim) with the low dimensional transformation
    of X
    P: is a numpy.ndarray of shape (n, n) containing the P affinities of X

    Returns: (dY, Q)
    dY is a numpy.ndarray of shape (n, n) containing the gradients of Y
    Q is a numpy.ndarray of shape (n, n) containing the Q affinities of Y
    """
