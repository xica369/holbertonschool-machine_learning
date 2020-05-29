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


def baum_welch(Observations, N, M, Transition=None, Emission=None, Initial=None):
    """
    The Baum-Welch Algorithm
    """
