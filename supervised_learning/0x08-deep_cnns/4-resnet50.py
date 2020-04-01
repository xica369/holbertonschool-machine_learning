#!/usr/bin/env python3

"""function that builds the ResNet-50 architecture:

You can assume the input data will have shape (224, 224, 3)
All convolutions inside and outside the blocks should be followed by batch
normalization along the channels axis and a rectified linear activation (ReLU)
All weights should use he normal initialization

Returns: the keras model"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ResNet-50"""

    X = K.Input(shape=(224, 224, 3))
    he_normal = K.initializers.he_normal()

    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=7,
                            padding="same",
                            kernel_initializer=he_normal,
                            strides=2)(X)

    batch_normalization = K.layers.BatchNormalization()(conv1)
    activation = K.layers.Activation("relu")(batch_normalization)

    max_pool1 = K.layers.MaxPooling2D(pool_size=3,
                                      strides=2,
                                      padding="same")(activation)

    filters = [64, 64, 256]
    strides = 1
    layer = projection_block(max_pool1, filters, strides)
    layer = identity_block(layer, filters)
    layer = identity_block(layer, filters)

    filters = [128, 128, 512]
    layer = projection_block(layer, filters)
    layer = identity_block(layer, filters)
    layer = identity_block(layer, filters)
    layer = identity_block(layer, filters)

    filters = [256, 256, 1024]
    layer = projection_block(layer, filters)
    layer = identity_block(layer, filters)
    layer = identity_block(layer, filters)
    layer = identity_block(layer, filters)
    layer = identity_block(layer, filters)
    layer = identity_block(layer, filters)

    filters = [512, 512, 2048]
    layer = projection_block(layer, filters)
    layer = identity_block(layer, filters)
    layer = identity_block(layer, filters)

    layer = K.layers.AveragePooling2D(pool_size=7,
                                      padding="same")(layer)

    layer = K.layers.Dense(units=1000,
                           activation="softmax")(layer)

    model = K.models.Model(inputs=X, outputs=layer)

    return model
