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

    layer3 = K.layers.Conv2D(filters=64,
                             kernel_size=1,
                             padding="same",
                             kernel_initializer=he_normal,
                             activation="relu")(layer2)

    layer4 = K.layers.Conv2D(filters=192,
                             kernel_size=3,
                             padding="same",
                             kernel_initializer=he_normal,
                             activation="relu")(layer3)

    layer5 = K.layers.MaxPooling2D(pool_size=3,
                                   strides=2,
                                   padding="same")(layer4)

    filters = [64, 96, 128, 16, 32, 32]
    layer6 = inception_block(layer5, filters)

    filters = [128, 128, 192, 32, 96, 64]
    layer7 = inception_block(layer6, filters)

    layer8 = K.layers.MaxPooling2D(pool_size=3,
                                   strides=2,
                                   padding="same")(layer7)

    filters = [192, 96, 208, 16, 48, 64]
    layer9 = inception_block(layer8, filters)

    filters = [160, 112, 224, 24, 64, 64]
    layer10 = inception_block(layer9, filters)

    filters = [128, 128, 256, 24, 64, 64]
    layer11 = inception_block(layer10, filters)

    filters = [112, 144, 288, 32, 64, 64]
    layer12 = inception_block(layer11, filters)

    filters = [256, 160, 320, 32, 128, 128]
    layer13 = inception_block(layer12, filters)

    layer14 = K.layers.MaxPooling2D(pool_size=3,
                                    strides=2,
                                    padding="same")(layer13)

    filters = [256, 160, 320, 32, 128, 128]
    layer15 = inception_block(layer14, filters)

    filters = [384, 192, 384, 48, 128, 128]
    layer16 = inception_block(layer15, filters)

    layer17 = K.layers.AveragePooling2D(pool_size=7,
                                        padding="same")(layer16)

    dropout = K.layers.Dropout(rate=0.4)(layer17)

    softmax = K.layers.Dense(units=1000,
                             activation="softmax")(dropout)

    model = K.models.Model(inputs=X, outputs=softmax)

    return model
