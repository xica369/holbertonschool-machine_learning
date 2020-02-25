#!/usr/bin/env python3

"""Accuracy"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """function that calculates the accuracy of a prediction:
    y: placeholder for the labels of the input data
    y_pred: tensor containing the network’s predictions
    Returns: tensor with the decimal accuracy of the prediction"""

    equal = tf.equal(y_pred, y)
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

    return accuracy
