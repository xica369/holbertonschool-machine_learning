#!/usr/bin/env python3

"""Specificity"""

import numpy as np


def specificity(confusion):
    """function that calculates the specificity
    for each class in a confusion matrix"""

    TP = np.diag(confusion)
    FN = confusion.sum(axis=1) - TP
    FP = confusion.sum(axis=0) - TP
    TN = confusion.sum() - (FP + FN + TP)

    specificity = TN / (TN + FP)

    return specificity
