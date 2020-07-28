#!/usr/bin/env python3

"""
Scaled Dot Product Attention
"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Function that calculates the scaled dot product attention:

    - Q is a tensor with its last two dimensions as (..., seq_len_q, dk)
      containing the query matrix
    - K is a tensor with its last two dimensions as (..., seq_len_v, dk)
      containing the key matrix
    - V is a tensor with its last two dimensions as (..., seq_len_v, dv)
      containing the value matrix
    - mask is a tensor that can be broadcast into (..., seq_len_q, seq_len_v)
      containing the optional mask, or defaulted to None
        if mask is not None, multiply -1e9 to the mask and add it to the
        scaled matrix multiplication
    The preceding dimensions of Q, K, and V are the same

    Returns: output, weights
    - outputa tensor with its last two dimensions as (..., seq_len_q, dv)
      containing the scaled dot product attention
    - weights a tensor with its last two dimensions as
    (..., seq_len_q, seq_len_v) containing the attention weights
    """
