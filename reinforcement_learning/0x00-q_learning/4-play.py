#!/usr/bin/env python3

"""
Play
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Function that
    has the trained agent play an episode:

    - env is the FrozenLakeEnv instance
    - Q is a numpy.ndarray containing the Q-table
    - max_steps is the maximum number of steps in the episode

    Returns: the total rewards for the episode
    """

    state = env.reset()

    for step in range(max_steps):
        # Show current state of environment on screen
        # Choose action with highest Q-value for current state
        # Take new action

        env.render()
        action = np.argmax(Q[state, :])
        state, reward, done, info = env.step(action)

        if done:
            env.render()
            break

    env.close()

    return reward
