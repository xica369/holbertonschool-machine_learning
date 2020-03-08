#!/usr/bin/env python3

"""F1 score"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """function that calculates the F1 score of a confusion matrix"""

    _sensitivity = sensitivity(confusion)
    _precision = precision(confusion)
    F1_score = 2 * (_precision * _sensitivity) / (_precision + _sensitivity)

    return F1_score
