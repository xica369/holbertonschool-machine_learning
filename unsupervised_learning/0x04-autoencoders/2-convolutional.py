#!/usr/bin/env python3

"""
Convolutional Autoencoder
"""

import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """
    Function that creates a convolutional autoencoder:

    input_dims: tuple of integers containing the dimensions of the model input
    filters: list containing the number of filters for each convolutional layer
    in the encoder, respectively
    latent_dims: tuple of integers containing the dimensions of the latent
    space representation

    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the full autoencoder model
    """

    he_normal = K.initializers.he_normal()

    # ========= ENCODER =========

    # create a placeholder to encoder
    X_encoder = x = K.Input(shape=input_dims)

    # encoder will consist in a stack of Conv2D and MaxPooling2D layers
    for filter in (filters):
        x = K.layers.Conv2D(filters=filter,
                            kernel_size=3,
                            padding="same",
                            kernel_initializer=he_normal,
                            activation="relu")(x)

        x = K.layers.MaxPool2D(pool_size=(2, 2), padding="same")(x)

    # create the encoder model
    encoder = K.models.Model(inputs=X_encoder, outputs=x)

    # ========= DECODER =========

    # create a placeholder to decoder
    X_decoder = x = K.Input(shape=latent_dims)

    # decoder will consist in a stack of Conv2D and UpSampling2D layers
    for iter in range(len(filters)-1, 0, -1):
        x = K.layers.Conv2D(filters=filters[iter],
                            kernel_size=3,
                            padding="same",
                            kernel_initializer=he_normal,
                            activation="relu")(x)

        x = K.layers.UpSampling2D(2)(x)

    x = K.layers.Conv2D(filters=filters[0],
                        kernel_size=3,
                        padding="valid",
                        kernel_initializer=he_normal,
                        activation="relu")(x)

    x = K.layers.UpSampling2D()(x)

    output = K.layers.Conv2D(filters=input_dims[2],
                             kernel_size=3,
                             padding="same",
                             kernel_initializer=he_normal,
                             activation="sigmoid")(x)

    # create the decoder model
    decoder = K.models.Model(inputs=X_decoder, outputs=output)

    # ========= AUTOENCODER =========

    # create a placeholder to autoencoder
    X_auto = K.Input(shape=input_dims)

    # get outputs of decoder to build the autoencoder model
    h = encoder(X_auto)
    Y = decoder(h)

    # create the autoencoder model and compile
    auto = K.models.Model(inputs=X_auto, outputs=Y)
    auto.compile(optimizer="Adam", loss="binary_crossentropy")

    return encoder, decoder, auto
