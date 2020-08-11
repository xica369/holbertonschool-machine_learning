#!/usr/bin/env python3

"""
Epsilon Greedy
"""


def epsilon_greedy(Q, state, epsilon):
    """
    Function that
    uses epsilon-greedy to determine the next action:

    - Q is a numpy.ndarray containing the q-table
    - state is the current state
    - epsilon is the epsilon to use for the calculation

    Returns: the next action index
    """
