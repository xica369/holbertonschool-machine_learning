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

    x = tf.layers.Dense(
        units=n,
        activation=activation,
        name='layer',
        dtype=tf.float32
    )

    mean, variance = tf.nn.moments(x(prev), axes=0, keep_dims=False)
<<<<<<< HEAD

    print(mean)
    print(variance)
=======
>>>>>>> 199267280ecb825231014868a1480f4db6efac95

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
