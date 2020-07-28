#!/usr/bin/env python3

"""
Positional Encoding
"""

import tensorflow as tf
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Function that calculates the positional encoding for a transformer:

    - max_seq_len is an integer representing the maximum sequence length
    - dm is the model depth

    Returns:
    a numpy.ndarray of shape (max_seq_len, dm) containing the positional
    encoding vectors
    """

    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]

    # positional_encoding = pos / (10000 ^ (2i/dmo))
    positional_encoding = pos / np.power(10000, (2 * (i//2)) / np.float32(dm))

    # sin to even indices in the array; 2i
    positional_encoding[:, 0::2] = np.sin(positional_encoding[:, 0::2])

    # cos to odd indices in the array; 2i+1
    positional_encoding[:, 1::2] = np.cos(positional_encoding[:, 1::2])

    return positional_encoding
