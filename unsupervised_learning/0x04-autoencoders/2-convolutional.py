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
