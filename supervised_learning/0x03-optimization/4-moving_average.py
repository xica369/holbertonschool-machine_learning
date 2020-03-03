#!/usr/bin/env python3

"""Moving Average"""

import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set:
    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Your moving average calculation should use bias correction
    Returns: a list containing the moving averages of data"""

    moving_average = []
    vt = 0

    for cont in range(len(data)):
        vt = beta * vt + (1 - beta) * data[cont]
        correct = vt / (1 - pow(beta, cont + 1))
        moving_average.append(correct)

    return moving_average
