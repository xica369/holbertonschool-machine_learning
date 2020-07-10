#!/usr/bin/env python3

"""
Function that creates the loss tensor and training op for the generator:

Z is the tf.placeholder that is the input for the generator
X is the tf.placeholder that is the input for the discriminator

Returns: loss, train_op
loss is the generator loss
train_op is the training operation for the generator
"""

import numpy as np
import tensorflow as tf
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_generator(Z):
    """
    Function that creates the loss tensor and training op for the generator
    """
