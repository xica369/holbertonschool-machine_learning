#!/usr/bin/env python3

"""function that builds the DenseNet-121 architecture:

growth_rate is the growth rate
compression is the compression factor
You can assume the input data will have shape (224, 224, 3)
All convolutions should be preceded by Batch Normalization and
a rectified linear activation (ReLU), respectively
All weights should use he normal initialization

Returns: the keras model"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """DenseNet-121"""

    X = K.Input(shape=(224, 224, 3))
    he_normal = K.initializers.he_normal()

    batch_normalization = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation("relu")(batch_normalization)

    conv = K.layers.Conv2D(filters=64,
                           kernel_size=7,
                           padding="same",
                           kernel_initializer=he_normal,
                           strides=2)(activation)

    max_pool = K.layers.MaxPooling2D(pool_size=3,
                                     strides=2,
                                     padding="same")(conv)

    denseBlock1, nb_filters = dense_block(max_pool, 64, growth_rate, 6)
    layer1, nb_filters = transition_layer(denseBlock1, nb_filters, compression)

    denseBlock2, nb_filters = dense_block(layer1, nb_filters, growth_rate, 12)
    layer2, nb_filters = transition_layer(denseBlock2, nb_filters, compression)

    denseBlock3, nb_filters = dense_block(layer2, nb_filters, growth_rate, 24)
    layer3, nb_filters = transition_layer(denseBlock3, nb_filters, compression)

    denseBlock4, nb_filters = dense_block(layer3, nb_filters, growth_rate, 16)

    average_pool = K.layers.AveragePooling2D(pool_size=7,
                                             padding="same")(denseBlock4)

    dense = K.layers.Dense(units=1000,
                           activation="softmax")(average_pool)

    model = K.models.Model(inputs=X, outputs=dense)

    return model
