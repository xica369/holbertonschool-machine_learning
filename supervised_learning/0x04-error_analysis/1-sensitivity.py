#!/usr/bin/env python3

"""Sensitivity"""

import numpy as np


def sensitivity(confusion):
    """function that calculates the sensitivity
    for each class in a confusion matrix"""

    TP = np.diag(confusion)
    FN = confusion.sum(axis=1) - TP

    sensitivity = TP / (TP + FN)

    return sensitivity
