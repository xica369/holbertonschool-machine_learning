#!/usr/bin/env python3

"""Accuracy"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """function that calculates the accuracy of a prediction:
    y: placeholder for the labels of the input data
    y_pred: tensor containing the network’s predictions
    Returns: tensor with the decimal accuracy of the prediction"""

    prediction = tf.argmax(y_pred, 1)
    correct_answer = tf.argmax(y, 1)
    equal = tf.equal(prediction, correct_answer)
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

    return accuracy
