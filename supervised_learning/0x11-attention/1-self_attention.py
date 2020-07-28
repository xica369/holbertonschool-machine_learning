#!/usr/bin/env python3

"""
class SelfAttention
inherits from tensorflow.keras.layers.Layer
to calculate the attention for machine translation
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    class SelfAttention
    """

    def __init__(self, units):
        """
        Class constructor
        - units is an integer representing the number of hidden units in
        the alignment model

        Public instance attributes:
        - W: a Dense layer with units units, to be applied to the previous
        decoder hidden state
        - U: a Dense layer with units units, to be applied to the encoder
        hidden states
        - V: a Dense layer with 1 units, to be applied to the tanh of the
        sum of the outputs of W and U
        """

    def call(self, s_prev, hidden_states):
        """
        Public instance method
        - s_prev is a tensor of shape (batch, units) containing the
        previous decoder hidden state
        - hidden_states is a tensor of shape (batch, input_seq_len, units)
        containing the outputs of the encoder

        Returns: context, weights
        - context is a tensor of shape (batch, units) that contains the
          context vector for the decoder
        - weights is a tensor of shape (batch, input_seq_len, 1) that
          contains the attention weights
        """
