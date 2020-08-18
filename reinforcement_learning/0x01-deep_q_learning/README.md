# 0x01. Deep Q-learning
Program that train an agent that can play Atari's Breakout:

## Requirements
- Ubuntu 16.04 LTS
- Python3 (version 3.5)
- numpy (version 1.15)
- gym (version 0.17.2)
- Tensorflow(version 1.14)
- keras (version 2.2.5)
- keras-rl (version 0.4.2)
- Pillow
- h5py

## How to use it
- First run the train.py file, which will create a file called policy.h5 with the weights of trained network. This may take time.
- Then run the play.py file, which will load the policy network saved in policy.h5 and display a game played by the agent trained by train.py
