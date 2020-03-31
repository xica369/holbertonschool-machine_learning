#!/usr/bin/env python3

"""function that builds the inception network:

You can assume the input data will have shape (224, 224, 3)
All convolutions inside and outside the inception block should use
a rectified linear activation (ReLU)

Returns: the keras model"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Inception Network"""

    X = K.Input(shape=(224, 224, 3))
    he_normal = K.initializers.he_normal()

    layer1 = K.layers.Conv2D(filters=64,
                             kernel_size=7,
                             padding="same",
                             kernel_initializer=he_normal,
                             activation="relu",
                             strides=2)(X)

    layer2 = K.layers.MaxPooling2D(pool_size=3,
                                   strides=2,
                                   padding="same")(layer1)

    layer3 = K.layers.Conv2D(filters=192,
                             kernel_size=3,
                             padding="same",
                             strides=1,
                             kernel_initializer=he_normal,
                             activation="relu")(layer2)

    layer4 = K.layers.MaxPooling2D(pool_size=3,
                                   strides=2,
                                   padding="same")(layer3)

    filters = [64, 96, 128, 16, 32, 32]
    layer5 = inception_block(layer4, filters)

    filters = [128, 128, 192, 32, 96, 64]
    layer6 = inception_block(layer5, filters)

    layer7 = K.layers.MaxPooling2D(pool_size=3,
                                   strides=2,
                                   padding="same")(layer6)

    filters = [192, 96, 208, 16, 48, 64]
    layer8 = inception_block(layer7, filters)

    filters = [160, 112, 224, 24, 64, 64]
    layer9 = inception_block(layer8, filters)

    filters = [128, 128, 256, 24, 64, 64]
    layer10 = inception_block(layer9, filters)

    filters = [112, 144, 288, 32, 64, 64]
    layer11 = inception_block(layer10, filters)

    filters = [256, 160, 320, 32, 128, 128]
    layer12 = inception_block(layer11, filters)

    layer13 = K.layers.MaxPooling2D(pool_size=3,
                                    strides=2,
                                    padding="same")(layer12)

    filters = [256, 160, 320, 32, 128, 128]
    layer14 = inception_block(layer13, filters)

    filters = [384, 192, 384, 48, 128, 128]
    layer15 = inception_block(layer14, filters)

    layer16 = K.layers.AveragePooling2D(pool_size=7,
                                        padding="same",
                                        strides=1)(layer15)

    dropout = K.layers.Dropout(rate=0.4)(layer16)

    softmax = K.layers.Dense(units=1000,
                             activation="softmax",
                             kernel_initializer=he_normal)(dropout)

    model = K.models.Model(inputs=X, outputs=softmax)

    return model
