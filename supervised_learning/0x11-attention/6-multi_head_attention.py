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
          dm is divisible by h
        - h is an integer representing the number of heads

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

        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = self.dm // self.h
        self.Wq = tf.keras.layers.Dense(self.dm)
        self.Wk = tf.keras.layers.Dense(self.dm)
        self.Wv = tf.keras.layers.Dense(self.dm)
        self.linear = tf.keras.layers.Dense(self.dm)

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

        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)

        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)

        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, weights = sdp_attention(q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        # (batch_size, seq_len_q, d_model)
        output = self.linear(concat_attention)

        return output, weights

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).

        Return:
        Tensor with the Transpose the result with shape
        (batch_size, num_heads, seq_len, depth)
        """

        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])
