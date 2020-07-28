#!/usr/bin/env python3

"""
class MultiHeadAttention
inherits from tensorflow.keras.layers.Layer
to perform multi head attention
"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi Head Attention
    """

    def __init__(self, dm, h):
        """
        Class constructor
        - dm is an integer representing the dimensionality of the model
        - h is an integer representing the number of heads
        - dm is divisible by h

        Public instance attributes:
        - h: the number of heads
        - dm: the dimensionality of the model
        - depth: the depth of each attention head
        - Wq: a Dense layer with dm units, used to generate the query matrix
        - Wk: a Dense layer with dm units, used to generate the key matrix
        - Wv: a Dense layer with dm units, used to generate the value matrix
        - linear: a Dense layer with dm units, used to generate the
          attention output
        """

    def call(self, Q, K, V, mask):
        """
        Public instance method
        - Q is a tensor of shape (batch, seq_len_q, dk) containing the
          input to generate the query matrix
        - K is a tensor of shape (batch, seq_len_v, dk) containing the
          input to generate the key matrix
        - V is a tensor of shape (batch, seq_len_v, dv) containing the
          input to generate the value matrix
        - mask is always None

        Returns: output, weights
        - output: tensor with its last two dimensions as (..., seq_len_q, dm)
          containing the scaled dot product attention
        - weights a tensor with its last three dimensions as
          (..., h, seq_len_q, seq_len_v) containing the attention weights
        """
