#!/usr/bin/env python3

"""Train_Op"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """Function that creates the training operation for the network:
    loss is the loss of the network’s prediction
    alpha is the learning rate
    Returns: an operation that trains the network using gradient descent"""

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train = optimizer.minimize(loss)

    return train
