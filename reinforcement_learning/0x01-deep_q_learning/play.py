#!/usr/bin/env python3

"""
script that load the policy network saved in policy.h5
and display a game played by the agent trained by train.py
"""

import gym
import numpy as np
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

build_model = __import__('train').build_model
InputProcessor = __import__('train').InputProcessor


if __name__ == '__main__':

    ENV_NAME = 'BreakoutDeterministic-v4'
    INPUT_SHAPE = (84, 84)
    WINDOW_LENGTH = 4

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)

    np.random.seed(3)
    env.seed(3)
    num_actions = env.action_space.n

    # build model
    model = build_model(INPUT_SHAPE, num_actions)

    # load wights of policy.h5
    model.load_weights("policy.h5")

    # where these experiences will be stored
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

    processor = InputProcessor()

    # use greedy policy - returns the current best action according to q_values
    policy = GreedyQPolicy()

    # Deep Q-Network and agent
    dqn = DQNAgent(model=model,
                   nb_actions=num_actions,
                   policy=policy,
                   memory=memory,
                   processor=processor,
                   nb_steps_warmup=50000,
                   gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)

    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    # Evaluate the algorithm for 6 episodes.
    dqn.test(env, nb_episodes=6, visualize=True)
