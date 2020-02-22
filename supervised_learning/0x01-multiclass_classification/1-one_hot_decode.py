#!/usr/bin/env python3

"""One-Hot Decode"""

import numpy as np


def one_hot_decode(one_hot):
    """that converts a one-hot matrix into a vector of labels:
    one_hot is a one-hot encoded numpy.ndarray with shape (classes, m)
    classes is the maximum number of classes
    m is the number of examples
    Returns: a numpy.ndarray with shape (m,) containing the numeric
    labels for each example, or None on failure"""

    if not isinstance(one_hot, np.ndarray):
        return None

    if len(one_hot) == 0:
        return None

    if np.where(one_hot == 1).all():
        return None

    Y = np.where(one_hot.T)
    Y = np.array(Y[1])

    return Y
