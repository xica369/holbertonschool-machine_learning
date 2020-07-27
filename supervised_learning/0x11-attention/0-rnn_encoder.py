#!/usr/bin/env python3

"""
RNN Encoder
"""

import tensorflow as tf


class RNNEncoder(tensorflow.keras.layers.Layer):
    """
    class RNNEncoder
    that inherits from tensorflow.keras.layers.Layer to encode for
    machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        - vocab is an integer representing the size of the input vocabulary
        - embedding is an integer representing the dimensionality of the
            embedding vector
        - units is an integer representing the number of hidden units in the
            RNN cell
        - batch is an integer representing the batch size

        Public instance attributes:
          - batch - the batch size
          - units - the number of hidden units in the RNN cell
          - embedding - a keras Embedding layer that converts words from the
              vocabulary into an embedding vector
          - gru - a keras GRU layer with units units

        Return both the full sequence of outputs as well as the last hidden
        state
        """


    def initialize_hidden_state(self):
        """
        Public instance method
        Initializes the hidden states for the RNN cell to a tensor of zeros

        Returns: a tensor of shape (batch, units)containing the initialized
        hidden states
        """


    def call(self, x, initial):
        """
        Public instance method
        - x is a tensor of shape (batch, input_seq_len) containing the input
          to the encoder layer as word indices within the vocabulary
        - initial is a tensor of shape (batch, units) containing the initial
          hidden state

        Returns: outputs, hidden
        - outputs is a tensor of shape (batch, input_seq_len, units)containing
        the outputs of the encoder
        - hidden is a tensor of shape (batch, units) containing the last hidden
        state of the encoder
