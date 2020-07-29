#!/usr/bin/env python3

"""
class EncoderBlock
inherits from tensorflow.keras.layers.Layer
to create an encoder block for a transformer
"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multi_head_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Transformer Encoder Block
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        - dm: the dimensionality of the model
        - h: the number of heads
        - hidden: the number of hidden units in the fully connected layer
        - drop_rate: the dropout rate

        Public instance attributes:
        - mha: a MultiHeadAttention layer
        - dense_hidden: the hidden dense layer with hidden units and relu
          activation
        - dense_output: the output dense layer with dm units
        - layernorm1: the first layer norm layer, with epsilon=1e-6
        - layernorm2: the second layer norm layer, with epsilon=1e-6
        - dropout1: the first dropout layer
        - dropout2: the second dropout layer
        """

        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        """
        - x: a tensor of shape (batch, input_seq_len, dm)containing the input
          to the encoder block
        - training: a boolean to determine if the model is training
        - mask: the mask to be applied for multi head attention

        Returns: a tensor of shape (batch, input_seq_len, dm) containing the
        block’s output
        """

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        output = self.dense_hidden(out1)
        output = self.dense_output(output)  # (batch_size, input_seq_len, d_model)
        output = self.dropout2(output, training=training)
        out2 = self.layernorm2(out1 + output)  # (batch_size, input_seq_len, d_model)

        return out2
