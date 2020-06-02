#!/usr/bin/env python3

"""
Function that performs the forward algorithm for a hidden markov model:

Observation is a numpy.ndarray of shape (T,) with the index of the observation
T is the number of observations
Emission is a numpy.ndarray of shape (N, M) containing the emission probability
of a specific observation given a hidden state
Emission[i, j] is the probability of observing j given the hidden state i
N is the number of hidden states
M is the number of all possible observations
Transition is 2D numpy.ndarray of shape (N,N) with the transition probabilities
Transition[i,j] the probability of transitioning from the hidden state i to j
Initial a numpy.ndarray of shape (N, 1) containing the probability of starting
in a particular hidden state

Returns: P, F, or None, None on failure
P is the likelihood of the observations given the model
F is a numpy.ndarray of shape (N, T) containing the forward path probabilities
F[i, j] is the probability of being in hidden state i at time j given the
previous observations
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    The Forward Algorithm
    """

    try:
        N = Emission.shape[0]

        if Observation.ndim != 1:
            return None, None

        if Emission.ndim != 2:
            return None, None

        if Transition.shape != (N, N):
            return None, None

        if Initial.shape != (N, 1):
            return None, None

        if not np.isclose(np.sum(Emission, axis=1), 1).all():
            return None, None

        if not np.isclose(np.sum(Transition, axis=1), 1).all():
            return None, None

        if not np.isclose(np.sum(Initial, axis=0), 1).all():
            return None, None

        T = Observation.shape[0]
        F = np.zeros((N, T))

        for idx, obs in enumerate(Observation):
            F[:, idx] = np.dot(F[:, idx-1], Transition[:, :])*Emission[:, obs]

            if idx == 0:
                F[:, idx] = Initial.T * Emission[:, obs]

        P = np.sum(F[:, -1])

        return P, F

    except Exception:
        return None, None
