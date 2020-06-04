#!/usr/bin/env python3

"""
Function that performs the Baum-Welch algorithm for a hidden markov model:

Observation is a numpy.ndarray of shape (T,) that contains the index of the
observation
T is the number of observations
N is the number of hidden states
M is the number of possible observations
Transition is the initialized transition probabilities, defaulted to None
Emission is the initialized emission probabilities, defaulted to None
Initial is the initiallized starting probabilities, defaulted to None
If Transition, Emission, or Initial is None, initialize the probabilities as
being a uniform distribution
Returns: the converged Transition, Emission, or None, None on failure
"""

import numpy as np


def baum_welch(Observations, N, M, Transition=None, Emission=None,
               Initial=None):
    """
    The Baum-Welch Algorithm
    """

    try:
        if Observation.ndim != 1:
            return None, None

        if not isinstance(N, int) or N < 1:
            return None, None

        if not isinstance(M, int) or M < 1:
            return None, None

        if Transition is None:
            Transition = np.ones((N, N)) / N

        if Emission is None:
            Emission = np.ones((N, M)) / M

        if Initial is None:
            Initial = np.ones((N, 1)) / N

        if not np.isclose(np.sum(Emission, axis=1), 1).all():
            return None, None

        if not np.isclose(np.sum(Transition, axis=1), 1).all():
            return None, None

        if not np.isclose(np.sum(Initial, axis=0), 1).all():
            return None, None

        if Emission.shape != (N, M):
            return None, None

        if Transition.shape != (N, N):
            return None, None

        if Initial.shape != (N, 1):
            return None, None

    except Exception:
        return None, None
