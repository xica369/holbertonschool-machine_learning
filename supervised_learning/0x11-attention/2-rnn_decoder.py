#!/usr/bin/env python3

"""
class RNNDecoder
inherits from tensorflow.keras.layers.Layer
to decode for machine translation
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNN Decoder
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        - vocab is an integer representing the size of the output vocabulary
        - embedding is an integer representing the dimensionality of the
        embedding vector
        - units is an integer representing the number of hidden units in the
        RNN cell
        - batch is an integer representing the batch size

        Public instance attributes:
        - embedding: a keras Embedding layer that converts words from the
        vocabulary into an embedding vector
        - gru: a keras GRU layer with units units

        """

    def call(self, x, s_prev, hidden_states):
        """
        Public instance method
        - x is a tensor of shape (batch, 1) containing the previous word in
          the target sequence as an index of the target vocabulary
        - s_prev is a tensor of shape (batch, units) containing the previous
          decoder hidden state
        - hidden_states is a tensor of shape (batch, input_seq_len, units)
          containing the outputs of the encoder

        Returns: y, s
        - y is a tensor of shape (batch, vocab) containing the output word
          as a one hot vector in the target vocabulary
        - s is a tensor of shape (batch, units) containing the new decoder
          hidden state
        """
