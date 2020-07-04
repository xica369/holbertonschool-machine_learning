#!/usr/bin/env python3

"""
LSTM Cell
"""

import numpy as np


class LSTMCell:
    """ class LSTMcell"""

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs

        Public instance attributes:
        Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
        that represent the weights and biases of the cell:

        Wf and bf are for the forget gate
        Wu and bu are for the update gate
        Wc and bc are for the intermediate cell state
        Wo and bo are for the output gate
        Wy and by are for the outputs
        """

        # weights
        self.Wf = np.random.normal(size=(i+h, h))
        self.Wu = np.random.normal(size=(i+h, h))
        self.Wc = np.random.normal(size=(i+h, h))
        self.Wo = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))

        # biases
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Public instance method that performs forward propagation for one
        time step

        x_t is a numpy.ndarray of shape (m, i) that contains the data input
        for the cell
          m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
        hidden state
        c_prev is a numpy.ndarray of shape (m, h) containing the previous
        cell state

        Returns: h_next, c_next, y
        h_next is the next hidden state
        c_next is the next cell state
        y is the output of the cell
        """

        Whx = np.concatenate((h_prev, x_t), axis=1)

        # forget gate = sigmoide(Wf[h_prev, x_t] + bf)
        f_t = self.sigmoide(np.dot(Whx, self.Wf) + self.bf)

        # update/input gate = sigmoide(Wu[h_prev, x_t] + bu)
        u_t = self.sigmoide(np.dot(Whx, self.Wu) + self.bu)

        # output gate = sigmoide(Wo[h_prev, x_t] + bo)
        o_t = self.sigmoide(np.dot(Whx, self.Wo) + self.bo)

        # candidate_value = tanh(Wc[h_prev, x_t] + bc)
        c_hat = np.tanh(np.dot(Whx, self.Wc) + self.bc)

        # update  candidate value
        c_next = f_t * c_prev + u_t * c_hat

        # update hidden state
        h_next = o_t * np.tanh(c_next)

        # output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y

    def softmax(self, X):
        """
        softmax function
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def sigmoide(self, X):
        """
        sigmoide function
        """
        return 1 / (1 + np.exp(-X))
