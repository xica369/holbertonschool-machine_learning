#!/usr/bin/env python3

"""function that builds an identity block:

A_prev is the output from the previous layer
filters is a tuple or list containing F11, F3, F12, respectively:
F11 is the number of filters in the first 1x1 convolution
F3 is the number of filters in the 3x3 convolution
F12 is the number of filters in the second 1x1 convolution
All convolutions inside the block should be followed by batch normalization
along the channels axis and a rectified linear activation (ReLU), respectively
All weights should use he normal initialization
Returns: the activated output of the identity block"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Identity Block"""

    he_normal = K.initializers.he_normal()

    layer1 = K.layers.Conv2D(filters=filters[0],
                             kernel_size=1,
                             padding="same",
                             kernel_initializer=he_normal,
                             activation="relu")(A_prev)

    batch_normalization = K.layers.BatchNormalization()(layer1)
    activation = K.layers.Activation("relu")(batch_normalization)

    layer2 = K.layers.Conv2D(filters=filters[1],
                             kernel_size=3,
                             kernel_initializer=he_normal,
                             padding="same",
                             activation="relu")(activation)

    batch_normalization = K.layers.BatchNormalization()(layer2)
    activation = K.layers.Activation("relu")(batch_normalization)

    layer3 = K.layers.Conv2D(filters=filters[2],
                             kernel_size=1,
                             padding="same",
                             kernel_initializer=he_normal,
                             activation="relu")(activation)

    batch_normalization = K.layers.BatchNormalization()(layer3)
    add = K.layers.Add()([batch_normalization, A_prev])
    activation = K.layers.Activation("relu")(add)

    return activation
