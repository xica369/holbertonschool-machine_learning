#!/usr/bin/env python3

"""
Function that performs forward propagation for a bidirectional RNN:

bi_cells is an instance of BidirectinalCell that will be used for the
forward propagation
X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
  t is the maximum number of time steps
  m is the batch size
  i is the dimensionality of the data
h_0 is the initial hidden state in the forward direction, given as a
numpy.ndarray of shape (m, h)
  h is the dimensionality of the hidden state
h_t is the initial hidden state in the backward direction, given as a
numpy.ndarray of shape (m, h)

Returns: H, Y
H is a numpy.ndarray containing all of the concatenated hidden states
Y is a numpy.ndarray containing all of the outputs
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Bidirectional RNN"""

    time_steps = X.shape[0]
    H_forward = np.array(([h_0]))
    H_forward = np.repeat(H_forward, time_steps, axis=0)
    H_backward = np.zeros(shape=H_forward.shape)

    hb = h_t
    hf = h_0
    tt = 0

    for t in range(time_steps - 1, -1, -1):

        # calculate h in forward direction
        hf = bi_cell.forward(hf, X[tt])
        H_forward[tt] = hf

        # calculate h in backward direction
        hb = bi_cell.backward(hb, X[t])
        H_backward[t] = hb

        tt += 1

    H = np.concatenate((H_forward, H_backward), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
