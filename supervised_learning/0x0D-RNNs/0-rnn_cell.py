#!/usr/bin/env python3

"""
class RNNCell that represents a cell of a simple RNN
"""

import numpy as np


class RNNCell:
    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs

        Public instance attributes Wh, Wy, bh, by that represent the weights
        and biases of the cell:
        Wh and bh are for the concatenated hidden state and input data
        Wy and by are for the output
        """

        # weights
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))

        # biases
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Instance method that performs forward propagation for one time step

        x_t is a numpy.ndarray of shape (m, i) that contains the data input
        forthe cell
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
        hidden state

        Returns: h_next, y
        h_next is the next hidden state
        y is the output of the cell
        """

        Wx = np.concatenate((h_prev, x_t), axis=1)

        # h_next = tanh(Wh[h_prev, x_t] + bh)
        h_next = np.tanh(np.dot(Wx, self.Wh) + self.bh)

        # y = softmax(Wy * h_next + by)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y

    def softmax(self, X):
        """
        softmax function
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
