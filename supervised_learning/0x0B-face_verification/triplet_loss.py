#!/usr/bin/env python3

"""Create a custom layer class TripletLoss"""

import tensorflow


class TripletLoss(tensorflow.keras.layers.Layer):
    """class TripletLoss that inherits from tensorflow.keras.layers.Layer"""

    def __init__(self, alpha, **kwargs):
        """Initialize Triplet Loss

        alpha is the alpha value used to calculate the triplet loss
        sets the public instance attribute alpha"""

        self.alpha = alpha
        super(TripletLoss, self).__init__(**kwargs)
