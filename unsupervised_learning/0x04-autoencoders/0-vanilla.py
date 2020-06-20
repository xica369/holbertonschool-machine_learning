#!/usr/bin/env python3

"""
Vanilla Autoencoder
"""

import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Function that creates an autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
    latent_dims is an integer containing the dimensions of the latent space
    representation

    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the full autoencoder model
    """

    # ========= ENCODER =========

    # create a placeholder to encoder
    X_encoder = K.Input(shape=(input_dims, ))
    input = X_encoder

    # create encoder's hidden layers
    for hidden_layer in hidden_layers:
        input = K.layers.Dense(hidden_layer, activation="relu")(input)

    # latent space representation
    h = K.layers.Dense(latent_dims, activation="relu")(input)

    # create the encoder model
    encoder = K.models.Model(inputs=X_encoder, outputs=h)

    # ========= DECODER =========

    # create a placeholder to decoder
    X_decoder = K.Input(shape=(latent_dims, ))
    input = X_decoder

    # create decoder's hidden layers
    for hidden_layer in reversed(hidden_layers):
        input = K.layers.Dense(hidden_layer, activation="relu")(input)

    output = K.layers.Dense(input_dims, activation="sigmoid")(input)

    # create the decoder model
    decoder = K.models.Model(inputs=X_decoder, outputs=output)

    # ========= AUTOENCODER =========

    # create a placeholder to autoencoder
    X_auto = K.Input(shape=(input_dims, ))

    # get outputs of decoder to build the autoencoder model
    h = encoder(X_auto)
    Y = decoder(h)

    # create the autoencoder model and compile
    auto = K.models.Model(inputs=X_auto, outputs=Y)
    auto.compile(optimizer="Adam", loss="binary_crossentropy")

    return encoder, decoder, auto
