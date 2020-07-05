#!/usr/bin/env python3

"""
Bidirectional Cell Forward
"""

import numpy as np


class BidirectionalCell:
    """class BidirectionalCell
    that represents a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden states
        o is the dimensionality of the outputs

        Public instance attributes Whf, Whb, Wy, bhf, bhb, by
        that represent the weights and biases of the cell

        Whf and bhfare for the hidden states in the forward direction
        Whb and bhbare for the hidden states in the backward direction
        Wy and byare for the outputs
        """

        # Weights
        self.Whf = np.random.normal(size=(i+h, h))
        self.Whb = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h+h, o))

        # Biases
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Public instance method that calculates the hidden state in the
        forward direction for one time step

        x_t is a numpy.ndarray of shape (m, i) that contains the data input
        for the cell
          m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
        hidden state

        Returns: h_next, the next hidden state
        """

        Wx = np.concatenate((h_prev, x_t), axis=1)

        # h_next = tanh(Wh_forward[h_prev, x_t] + bh_forward)
        h_next = np.tanh(np.dot(Wx, self.Whf) + self.bhf)

        return h_next
