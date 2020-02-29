#!/usr/bin/env python3

"""Adam Upgraded"""

import numpy as np
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """training operation for a neural network with
    the Adam optimization algorithm:

    loss is the loss of the network
    alpha is the learning rate
    beta1 is the weight used for the first moment
    beta2 is the weight used for the second moment
    epsilon is a small number to avoid division by zero
    Returns: the Adam optimization operation"""
