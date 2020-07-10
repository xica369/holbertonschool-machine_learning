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
