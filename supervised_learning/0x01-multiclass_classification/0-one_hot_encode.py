#!/usr/bin/env python3

"""One-Hot Encode"""

import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix
    Y is a numpy.ndarray with shape (m,) containing numeric class labels
    m is the number of examples
    classes is the maximum number of classes found in Y
    Returns: a one-hot encoding of Y with shape (classes, m),
    or None on failure"""

    if not isinstance(Y, np.ndarray):
        return None

    if len(Y) == 0:
        return None

    if not isinstance(classes, int):
        return None

    if classes <= Y.max():
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    colum = np.arange(m)
    one_hot[Y, colum] = 1

    return one_hot
