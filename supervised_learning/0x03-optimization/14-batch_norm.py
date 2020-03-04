#!/usr/bin/env python3

"""Batch Normalization Upgraded"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """batch normalization layer for a neural network in tensorflow:

    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation: activation function that should be used
    on the output of the layer
    Returns: a tensor of the activated output for the layer"""

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    x = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        name='layer'
    )

    mean, variance = tf.nn.moments(x(prev), axes=0)

    tensor = tf.nn.batch_normalization(
        x,
        mean,
        variance,
        offset=None,
        scale=None,
        variance_epsilon=1e-8,
        name=None
    )

    return tensor
