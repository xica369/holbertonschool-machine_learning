#!/usr/bin/env python3

"""functin that builds an inception block:

A_prev is the output from the previous layer
filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP, respectively:
F1 is the number of filters in the 1x1 convolution
F3R is the number of filters in the 1x1 convolution before the 3x3 convolution
F3 is the number of filters in the 3x3 convolution
F5R is the number of filters in the 1x1 convolution before the 5x5 convolution
F5 is the number of filters in the 5x5 convolution
FPP is the number of filters in the 1x1 convolution after the max pooling
All convolutions inside the inception block should use
a rectified linear activation (ReLU)
Returns: the concatenated output of the inception block"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Inception Block"""

    layers_list = []
    he_normal = K.initializers.he_normal()

    layer_F1 = K.layers.Conv2D(filters=filters[0],
                               kernel_size=1,
                               activation="relu",
                               padding="same",
                               kernel_initializer=he_normal)(A_prev)
    layers_list.append(layer_F1)

    layer_F3R = K.layers.Conv2D(filters=filters[1],
                                kernel_size=1,
                                activation="relu",
                                padding="same",
                                kernel_initializer=he_normal)(A_prev)

    layer_F3 = K.layers.Conv2D(filters=filters[2],
                               kernel_size=3,
                               activation="relu",
                               padding="same",
                               kernel_initializer=he_normal)(layer_F3R)
    layers_list.append(layer_F3)

    layer_F5R = K.layers.Conv2D(filters=filters[3],
                                kernel_size=1,
                                activation="relu",
                                padding="same",
                                kernel_initializer=he_normal)(A_prev)

    layer_F5 = K.layers.Conv2D(filters=filters[4],
                               kernel_size=5,
                               activation="relu",
                               padding="same",
                               kernel_initializer=he_normal)(layer_F5R)
    layers_list.append(layer_F5)

    layer_max_pool = K.layers.MaxPooling2D(pool_size=3,
                                           padding="same",
                                           strides=1)(A_prev)

    layer_FPP = K.layers.Conv2D(filters=filters[5],
                                kernel_size=1,
                                activation="relu",
                                padding="same",
                                kernel_initializer=he_normal)(layer_max_pool)
    layers_list.append(layer_FPP)

    concatenate = K.layers.concatenate(layers_list, axis=-1)
    return concatenate
