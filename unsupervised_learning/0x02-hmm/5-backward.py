#!/usr/bin/env python3

"""
Function that performs the backward algorithm for a hidden markov model:

Observation is a numpy.ndarray of shape (T,) that contains the index of the
observation
T is the number of observations
Emission is a numpy.ndarray of shape (N, M) containing the emission probability
of a specific observation given a hidden state
Emission[i, j] is the probability of observing j given the hidden state i
N is the number of hidden states
M is the number of all possible observations
Transition is a 2D numpy.ndarray of shape (N, N) containing the transition
probabilities
Transition[i, j] is the probability of transitioning from the hidden state
i to j
Initial a numpy.ndarray of shape (N, 1) containing the probability of starting
in a particular hidden state

Returns: P, B, or None, None on failure
Pis the likelihood of the observations given the model
B is a numpy.ndarray of shape (N, T) containing the backward path probabilities
B[i, j] is the probability of generating the future observations from hidden
state i at time j
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    The Backward Algorithm
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
        B = np.ones((N, T))

        for obs in reversed(range(T-1)):
            for h_state in range(N):
                B[h_state, obs] = (np.sum(B[:, obs + 1] *
                                          Transition[h_state, :] *
                                          Emission[:, Observation[obs + 1]]))

        P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

        return P, B

    except Exception:
        return None, None
