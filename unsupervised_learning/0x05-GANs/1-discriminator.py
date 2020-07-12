#!/usr/bin/env python3

"""
Function that creates a discriminator network for MNIST digits:

X is a tf.tensor containing the input to the discriminator network

Returns Y, a tf.tensor containing the classification made by the discriminator
"""

import numpy as np
import tensorflow as tf


def discriminator(X):
    """
    Function that creates a discriminator network for MNIST digits
    """

    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):

        layer_1 = tf.layers.Dense(128,
                                  activation="relu",
                                  name="layer_1")(X)

        layer_2 = tf.layers.Dense(1,
                                  activation="sigmoid",
                                  name="layer_2")(layer_1)

    return layer_2
