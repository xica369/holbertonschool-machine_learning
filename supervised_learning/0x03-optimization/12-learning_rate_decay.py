#!/usr/bin/env python3

"""Learning Rate Decay Upgraded"""

import numpy as np
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """learning rate decay operation in tensorflow using inverse time decay:

    alpha is the original learning rate
    decay_rate: weight to determine the rate at which alpha will decay
    global_step: number of passes of gradient descent that have elapsed
    decay_step: numb of passes of gradient descent that should occur
    before alpha is decayed further
    the learning rate decay should occur in a stepwise fashion
    Returns: the learning rate decay operation"""
