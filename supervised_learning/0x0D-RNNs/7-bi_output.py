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

        Whx = np.concatenate((h_prev, x_t), axis=1)

        # h_next = tanh(Wh_forward[h_prev, x_t] + bh_forward)
        h_next = np.tanh(np.dot(Whx, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Public instance method that calculates the hidden state in the
        backward direction for one time step

        x_t is a numpy.ndarray of shape (m, i) that contains the data input
        for the cell
          m is the batch size for the data
        h_next is a numpy.ndarray of shape (m, h) containing the next
        hidden state

        Returns: h_pev, the previous hidden state
        """

        Whx = np.concatenate((h_next, x_t), axis=1)

        # h_next = tanh(Wh_backward[h_next, x_t] + bh_backward)
        h_next = np.tanh(np.dot(Whx, self.Whb) + self.bhb)

        return h_next

    def output(self, H):
        """
        Public instance method that calculates all outputs for the RNN:

        H is a numpy.ndarray of shape (t, m, 2 * h) that contains the
        concatenated hidden states from both directions, excluding their
        initialized states
          t is the number of time steps
          m is the batch size for the data
          h is the dimensionality of the hidden states

        Returns: Y, the outputs
        """

        for t in range(H.shape[0]):
            y_hat = self.softmax(np.dot(H[t], self.Wy) + self.by)

            if t == 0:
                Y = np.array([y_hat])
            else:
                Y = np.append(Y, [y_hat], axis=0)

        return Y

    def softmax(self, X):
        """
        softmax function
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
