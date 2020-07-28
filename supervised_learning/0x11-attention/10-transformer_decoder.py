#!/usr/bin/env python3

"""
class Decoder
inherits from tensorflow.keras.layers.Layer
to create the decoder for a transformer
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Transformer Decoder
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor
        - N: the number of blocks in the encoder
        - dm: the dimensionality of the model
        - h: the number of heads
        - hidden: the number of hidden units in the fully connected layer
        - target_vocab: the size of the target vocabulary
        - max_seq_len: the maximum sequence length possible
        - drop_rate: the dropout rate

        Public instance attributes:
        - N: the number of blocks in the encoder
        - dm: the dimensionality of the model
        - embedding: the embedding layer for the targets
        - positional_encoding: a numpy.ndarray of shape (max_seq_len, dm)
          containing the positional encodings
        - blocks: a list of length N containing all of the DecoderBlock‘s
        - dropout: the dropout layer, to be applied to the positional encodings
        """

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Public instance method
        - x: a tensor of shape (batch, target_seq_len, dm)containing the
          input to the decoder
        - encoder_output - a tensor of shape (batch, input_seq_len, dm)
          containing the output of the encoder
        - training - a boolean to determine if the model is training
        - look_ahead_mask - the mask to be applied to the first multi head
          attention layer
        - padding_mask: the mask to be applied to the second multi head
          attention layer

        Returns:
        a tensor of shape (batch, target_seq_len, dm) containing the
        decoder output
        """