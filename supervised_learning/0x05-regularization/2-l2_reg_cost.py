#!/usr/bin/env python3

"""L2 Regularization Cost"""

import tensorflow as tf


def l2_reg_cost(cost):
    """calculates the cost of a neural network with L2 regularization:

    cost is a tensor with the cost of the network without L2 regularization
    Returns: a tensor with the cost of the network"""

    return cost + tf.losses.get_regularization_losses(scope=None)
