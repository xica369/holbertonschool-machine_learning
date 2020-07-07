#!/usr/bin/env python3

"""
Function that creates input for the generator:

m is the number of samples that should be generated
n is the number of dimensions of each sample

Returns: Z, a numpy.ndarray of shape (m, n) containing the uniform samples
"""

import numpy as np
import tensorflow as tf


def sample_Z(m, n):
    """
    Function that creates input for the generator
    """
