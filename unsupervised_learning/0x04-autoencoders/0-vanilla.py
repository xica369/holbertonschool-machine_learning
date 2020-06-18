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

    X_encoder = K.Input(shape=(input_dims, ))
    X_decoder = K.Input(shape=(latent_dims, ))

    input = X_encoder
    for hidden_layer in hidden_layers:
        input = K.layers.Dense(hidden_layer, activation="relu")(input)

    h = K.layers.Dense(latent_dims, activation="relu")(input)

    input = X_decoder
    for hidden_layer in reversed(hidden_layers):
        input = K.layers.Dense(hidden_layer, activation="relu")(input)

    output = K.layers.Dense(input_dims, activation="sigmoid")(input)

    encoder = K.models.Model(inputs=X_encoder, outputs=h)
    decoder = K.models.Model(inputs=X_decoder, outputs=output)
    auto = K.models.Model(inputs=X_encoder, outputs=output)

    auto.compile(optimizer="Adam", loss="binary_crossentropy")

    return encoder, decoder, auto
