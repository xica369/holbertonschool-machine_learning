#!/usr/bin/env python3

"""Batch Normalization Upgraded"""

import numpy as np
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """batch normalization layer for a neural network in tensorflow:

    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation: activation function that should be used
    on the output of the layer
    Returns: a tensor of the activated output for the layer"""
