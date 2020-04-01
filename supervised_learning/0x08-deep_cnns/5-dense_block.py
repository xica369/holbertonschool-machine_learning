#!/usr/bin/env python3

"""function that builds a dense block:

X is the output from the previous layer
nb_filters is an integer representing the number of filters in X
growth_rate is the growth rate for the dense block
layers is the number of layers in the dense block
You should use the bottleneck layers used for DenseNet-B
All weights should use he normal initialization
All convolutions should be preceded by Batch Normalization and
a rectified linear activation (ReLU), respectively

Returns: The concatenated output of each layer within the Dense Block and
the number of filters within the concatenated outputs, respectively"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Dense Block"""

    he_normal = K.initializers.he_normal()
    concatenate = X

    for layer in range(layers):
        nb_filters = nb_filters + growth_rate

        batch_normalization = K.layers.BatchNormalization()(concatenate)
        activation = K.layers.Activation("relu")(batch_normalization)
        conv = K.layers.Conv2D(filters=4*growth_rate,
                               kernel_size=1,
                               padding="same",
                               kernel_initializer=he_normal)(activation)

        batch_normalization = K.layers.BatchNormalization()(conv)
        activation = K.layers.Activation("relu")(batch_normalization)
        conv = K.layers.Conv2D(filters=growth_rate,
                               kernel_size=3,
                               padding="same",
                               kernel_initializer=he_normal)(activation)

        concatenate = K.layers.concatenate([concatenate, conv])

    return concatenate, nb_filters
