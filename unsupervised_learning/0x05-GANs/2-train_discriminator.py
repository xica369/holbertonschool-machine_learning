#!/usr/bin/env python3

"""
Function that creates the loss tensor and training op for the discriminator:

Z is the tf.placeholder that is the input for the generator
X is the tf.placeholder that is the real input for the discriminator

Returns: loss, train_op
  loss is the discriminator loss
  train_op is the training operation for the discriminator
"""

import numpy as np
import tensorflow as tf
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_discriminator(Z, X):
    """
    Function that creates the loss tensor and training op for the discriminator
    """
