#!/usr/bin/env python3

"""
Play
"""


def play(env, Q, max_steps=100):
    """
    Function that
    has the trained agent play an episode:

    - env is the FrozenLakeEnv instance
    - Q is a numpy.ndarray containing the Q-table
    - max_steps is the maximum number of steps in the episode

    Returns: the total rewards for the episode
    """
