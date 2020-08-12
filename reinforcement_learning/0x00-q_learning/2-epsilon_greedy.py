#!/usr/bin/env python3

"""
Epsilon Greedy
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Function that
    uses epsilon-greedy to determine the next action:

    - Q is a numpy.ndarray containing the q-table
    - state is the current state
    - epsilon is the epsilon to use for the calculation

    Returns: the next action index
    """

    if np.random.uniform(0, 1) < epsilon:

        # Explore: select a random action
        next_action = np.random.randint(Q.shape[1])

    else:

        # Exploit: select the action with max value (future reward)
        next_action = np.argmax(Q[state, :])

    return next_action
