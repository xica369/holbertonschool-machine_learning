#!/usr/bin/env python3

"""function that builds a projection block:

A_prev is the output from the previous layer
filters is a tuple or list containing F11, F3, F12, respectively:
F11 is the number of filters in the first 1x1 convolution
F3 is the number of filters in the 3x3 convolution
F12 is the number of filters in the second 1x1 convolution as well as
the 1x1 convolution in the shortcut connection
s is the stride of the first convolution in both the main path and
the shortcut connection
All convolutions inside the block should be followed by batch normalization
along the channels axis and a rectified linear activation (ReLU), respectively
All weights should use he normal initialization
Returns: the activated output of the projection block"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Projection Block"""

    he_normal = K.initializers.he_normal()

    layer1 = K.layers.Conv2D(filters=filters[0],
                             kernel_size=1,
                             padding="same",
                             kernel_initializer=he_normal,
                             strides=s)(A_prev)

    batch_normalization = K.layers.BatchNormalization()(layer1)
    activation = K.layers.Activation("relu")(batch_normalization)

    layer2 = K.layers.Conv2D(filters=filters[1],
                             kernel_size=3,
                             padding="same",
                             kernel_initializer=he_normal)(activation)

    batch_normalization = K.layers.BatchNormalization()(layer2)
    activation = K.layers.Activation("relu")(batch_normalization)

    layer3 = K.layers.Conv2D(filters=filters[2],
                             kernel_size=1,
                             padding="same",
                             kernel_initializer=he_normal)(activation)

    connect_layer = K.layers.Conv2D(filters=filters[2],
                                    kernel_size=1,
                                    padding="same",
                                    kernel_initializer=he_normal,
                                    strides=s)(A_prev)

    batch_normalization_l3 = K.layers.BatchNormalization()(layer3)
    batch_normalization_connect = K.layers.BatchNormalization()(connect_layer)
    add = K.layers.Add()([batch_normalization_l3, batch_normalization_connect])

    activation = K.layers.Activation("relu")(add)

    return activation
