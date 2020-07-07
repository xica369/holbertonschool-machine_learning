#!/usr/bin/env python3

"""
Function that creates a simple generator network for MNIST digits:

Z is a tf.tensor containing the input to the generator network

Returns X, a tf.tensor containing the generated image
"""

import numpy as np
import tensorflow as tf


def generator(Z):
    """
    Function that creates a simple generator network for MNIST digits
    """
