#!/usr/bin/env python3

"""Create Confusion"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """function that creates a confusion matrix"""

    cm = np.dot(labels.T, logits)

    return cm
