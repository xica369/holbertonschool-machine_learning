#!/usr/bin/env python3

"""
Function  that calculates the most likely sequence of hidden states for a
hidden markov model:

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
Transition[i, j] is the probability of transitioning from the hidden
state i to j
Initial a numpy.ndarray of shape (N, 1) containing the probability of starting
in a particular hidden state

Returns: path, P, or None, None on failure
path is the a list of length T containing the most likely sequence of hidden
states
P is the probability of obtaining the path sequence
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    The Viretbi Algorithm
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
        prev = np.zeros((N, T))

        # Initilaize the tracking tables from first observation
        F[:, 0] = Initial.T * Emission[:, Observation[0]]
        prev[:, 0] = 0

        # Iterate throught the observations updating the tracking tables
        for idx, obs in enumerate(Observation):
            if idx != 0:
                F[:, idx] = np.max(F[:, idx - 1] * Transition.T *
                                   Emission[np.newaxis, :, obs].T, 1)
                prev[:, idx] = np.argmax(F[:, idx - 1] * Transition.T, 1)

        # Build the output, optimal model trajectory (path)
        path = T * [1]
        path[-1] = np.argmax(F[:, T - 1])
        for idx in reversed(range(1, T)):
            path[idx - 1] = int(prev[path[idx], idx])

        # calculate the probability of obtaining the path sequence
        P = np.amax(F, axis=0)
        P = np.amin(P)

        return path, P

    except Exception:
        return None, None
