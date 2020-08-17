#!/usr/bin/env python3

"""
Script to train an agent that can play Atari’s Breakout
"""

import gym
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


class AtariBreakoutEnvManager():
    """
    This class will manage our Atari Breakout enviroment
    """

    def __init__(self):
        """
        class constructor
        """
        self.env = gym.make("Breakout-v0")
        self.env.reset()
        self.current_screen = None
        self.donde = False

    # Wrapped Functions

    def reset(self):
        """
        environment is reset to a starting state
        returns an initial observation from the environment.
        """
        self.env.reset()
        self.current_screen = None

    def close(self):
        """
        close the environment
        """
        self.env.close()

    def render(self, mode='human'):
        """
        render the current state to the screen.
        """
        return self.env.render(mode)

    def num_actions_available(self):
        """
        returns the number of actions available to an agent in the
        environment
        """
        return self.env.action_space.n

    def take_action(self, action):
        """
        Taking An Action In The Environment

        call step() on the environment, which will execute the given
        action taken by the agent in the environment.

        Return:
        the reward obtained
        """
        _, reward, self.done, _ = self.env.step(action.item())

        return reward

    def just_starting(self):
        """
        Starting An Episode

        returns True when the current_screen is None and
        returns False otherwise
        """
        return self.current_screen is None

