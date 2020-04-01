#!/usr/bin/env python3

"""function that builds a transition layer:

X is the output from the previous layer
nb_filters is an integer representing the number of filters in X
compression is the compression factor for the transition layer
Your code should implement compression as used in DenseNet-C
All weights should use he normal initialization
All convolutions should be preceded by Batch Normalization and
a rectified linear activation (ReLU), respectively

Returns: The output of the transition layer and the number of filters
within the output, respectively"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Transition Layer"""

    he_normal = K.initializers.he_normal()

    batch_normalization = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation("relu")(batch_normalization)

    conv = K.layers.Conv2D(filters=int(nb_filters*compression),
                           kernel_size=1,
                           padding="same",
                           kernel_initializer=he_normal)(activation)

    pool = K.layers.AveragePooling2D(pool_size=2,
                                     strides=2)(conv)

    nb_filters = int(nb_filters * compression)

    return pool, nb_filters
