#!/usr/bin/env python3

"""F1 score"""

import numpy as np


def f1_score(confusion):
    """function that calculates the F1 score of a confusion matrix"""

    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - TP
    FN = confusion.sum(axis=1) - TP
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1_score = 2 * (precision * recall) / (precision + recall)

    return F1_score
