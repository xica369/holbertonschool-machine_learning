#!/usr/bin/env python3

"""Create a Layer with L2 Regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a tensorflow layer that includes L2 regularization:

    prev is a tensor containing the output of the previous layer
    n is the number of nodes the new layer should contain
    activation is the activation function that should be used on the layer
    lambtha is the L2 regularization parameter
    Returns: the output of the new layer"""

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(scale=lambtha)

    layer = tf.layers.dense(
        inputs=prev,
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=regularizer
    )

    return layer
