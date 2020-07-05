#!/usr/bin/env python3

"""
Function that performs forward propagation for a deep RNN:

rnn_cells is a list of RNNCell instances of length l that will be used for the
forward propagation

l is the number of layers
X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
t is the maximum number of time steps
m is the batch size
i is the dimensionality of the data
h_0 is the initial hidden state, given as a numpy.ndarray of shape (l, m, h)
h is the dimensionality of the hidden state

Returns: H, Y
H is a numpy.ndarray containing all of the hidden states
Y is a numpy.ndarray containing all of the outputs
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Deep RNN
    """

    h_prev = h_0
    H = np.array(([h_prev]))
    H = np.repeat(H, X.shape[0] + 1, axis=0)

    for t in range(X.shape[0]):
        for layer, cell in enumerate(rnn_cells):
            if layer == 0:
                # call function with h_t and X_t of current time
                h_prev, y = cell.forward(H[t, layer], X[t])
            else:
                # call function with h_t and h_prev of current time
                h_prev, y = cell.forward(H[t, layer], h_prev)

            # update matrix of hidden states
            H[t + 1, layer] = h_prev

            # update matrix of outputs
            if t == 0:
                Y = np.array(([y]))
                Y = np.repeat(Y, X.shape[0], axis=0)

            else:
                Y[t] = y

    return H, Y
