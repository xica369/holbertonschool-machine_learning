#!/usr/bin/env python3

"""Create a custom layer class TripletLoss"""

import tensorflow
import tensorflow as tf


class TripletLoss(tensorflow.keras.layers.Layer):
    """class TripletLoss that inherits from tensorflow.keras.layers.Layer"""

    def __init__(self, alpha, **kwargs):
        """Initialize Triplet Loss

        alpha is the alpha value used to calculate the triplet loss
        sets the public instance attribute alpha"""

        self.alpha = alpha
        super(TripletLoss, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        """method that calculate Triplet Loss:
        inputs is a list containing the anchor, positive and
        negative output tensors from the last layer of the model, respectively
        Returns: a tensor containing the triplet loss values"""

        positive_output = inputs[1]
        negative_output = inputs[2]
        anchor_output = inputs[0]

        d_posit = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
        d_negat = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)

        margin = self.alpha
        loss = tf.maximum(margin + d_posit - d_negat, 0)

        return loss
