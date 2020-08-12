#!/usr/bin/env python3

"""
Q-learning
"""

import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Function that
    performs Q-learning:

    - env is the FrozenLakeEnv instance
    - Q is a numpy.ndarray containing the Q-table
    - episodes is the total number of episodes to train over
    - max_steps is the maximum number of steps per episode
    - alpha is the learning rate
    - gamma is the discount rate
    - epsilon is the initial threshold for epsilon greedy
    - min_epsilon is the minimum value that epsilon should decay to
    - epsilon_decay is the decay rate for updating epsilon between episodes

    Returns: Q, total_rewards
    - Q is the updated Q-table
    - total_rewards is a list containing the rewards per episode
    """

    max_epsilon = 1
    rewards_all_episodes = []

    # Q-learning algorithm
    for episode in range(episodes):

        # initialize new episode params
        state = env.reset()
        done = False
        rewards_current_episode = 0

        for step in range(max_steps):
            # Exploration-exploitation trade-off
            # Take new action
            # Update Q-table
            # Set new state
            # Add new reward

            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)

            # the agent falls in a hole
            if reward == 0 and done is True:
                reward = -1

            # Update Q-table for Q(s,a)
            Q[state, action] = (Q[state, action] * (1 - alpha) + alpha *
                                (reward + gamma * np.max(Q[new_state, :])))

            state = new_state
            rewards_current_episode += reward

            if done is True:
                break

        # Exploration rate decay
        epsilon = (min_epsilon + (max_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay*episode))

        rewards_all_episodes.append(rewards_current_episode)

    return Q, rewards_all_episodes


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
