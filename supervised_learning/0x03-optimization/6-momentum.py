#!/usr/bin/env python3

"""Momentum Upgraded"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """training operation for a neural network with the
    gradient descent and momentum optimization algorithm:
    loss is the loss of the network
    alpha is the learning rate
    beta1 is the momentum weight
    Returns: the momentum optimization operation"""

    optimization = tf.train.MomentumOptimizer(learning_rate=alpha,
                                              momentum=beta1)

    return optimization.minimize(loss)
