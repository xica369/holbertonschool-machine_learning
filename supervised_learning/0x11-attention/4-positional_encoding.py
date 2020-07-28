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
