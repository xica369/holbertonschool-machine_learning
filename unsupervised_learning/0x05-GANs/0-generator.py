#!/usr/bin/env python3

"""
Function that creates a simple generator network for MNIST digits:

Z is a tf.tensor containing the input to the generator network

Returns X, a tf.tensor containing the generated image
"""

import tensorflow as tf


def generator(Z):
    """
    Function that creates a simple generator network for MNIST digits
    """

    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):

        layer_1 = tf.layers.Dense(128,
                                  activation="relu",
                                  name="layer_1")(Z)

        layer_2 = tf.layers.Dense(784,
                                  activation="sigmoid",
                                  name="layer_2")(layer_1)


    return layer_2
