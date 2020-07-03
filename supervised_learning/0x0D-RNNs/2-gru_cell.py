#!/usr/bin/env python3

"""
GRU Cell
"""

import numpy as np


class GRUCell:
    """Class GRUE cell"""

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs

        Public instance attributes:
        Wz, Wr, Wh, Wy, bz, br, bh, by
        that represent the weights and biases of the cell

        Wz and bz are for the update gate
        Wr and br are for the reset gate
        Wh and bh are for the intermediate hidden state
        Wy and by are for the output
        """

        # weights
        self.Wz = np.random.normal(size=(i+h, h))
        self.Wr = np.random.normal(size=(i+h, h))
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))

        # bias
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Public instance method that performs forward propagation for one
        time step
        x_t is a numpy.ndarray of shape (m, i) that contains the data input
        for the cell
          m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
        hidden state

        Returns: h_next, y
        h_next is the next hidden state
        y is the output of the cell
        """

        Wcx = np.concatenate((h_prev, x_t), axis=1)
        Gu = self.softmax(np.dot(Wcx, self.Wz) + self.bz)
        Gr = self.softmax(np.dot(Wcx, self.Wr) + self.br)
        Wxc = np.concatenate(((h_prev * Gr), x_t), axis=1)
        C_hat = np.tanh(np.dot(Wcx, self.Wh) + self.bh)
        h_next = Gu * C_hat + (1 - Gu) + h_prev

        return h_next, h_next

    def softmax(self, X):
        """
        softmax function
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
