#!/usr/bin/env python3

"""Precision"""

import numpy as np


def precision(confusion):
    """function that calculates the precision
    for each class in a confusion matrix"""

    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - TP

    precision = TP / (TP + FP)

    return precision
