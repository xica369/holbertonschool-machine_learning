#!/usr/bin/env python3

"""calculates the definiteness of a matrix"""

import numpy as np


def definiteness(matrix):
    """
    matrix is a numpy.ndarray of shape (n, n) whose definiteness
    should be calculated

    Return: the string Positive definite, Positive semi-definite,
    Negative semi-definite, Negative definite, or Indefinite
    if the matrix is positive definite, positive semi-definite,
    negative semi-definite, negative definite of indefinite, respectively
    If matrix does not fit any of the above categories, return None"""

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.size == 0:
        return None

    n = len(matrix)
    if matrix.shape != (n, n):
        return None

    # check if matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    w, _ = np.linalg.eig(matrix)

    if np.all(w > 0):
        return "Positive definite"

    if np.all(w < 0):
        return "Negative definite"

    if np.any(w > 0) and np.any(w == 0):
        return "Positive semi-definite"

    if np.any(w < 0) and np.any(w == 0):
        return "Negative semi-definite"

    if not(np.any(w > 0) and np.any(w < 0) and np.any(w == 0)):
        return "Indefinite"

    else:
        return None
