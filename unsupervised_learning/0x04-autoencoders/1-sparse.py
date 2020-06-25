#!/usr/bin/env python3

"""
Sparse Autoencoder
"""

import tensorflow.keras as K


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Function that creates a sparse autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
    latent_dims is an integer containing the dimensions of the latent space
    representation
    lambtha is the regularization parameter used for L1 regularization on the
    encoded output

    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the sparse autoencoder model
    """

    input_encoder = K.Input(shape=(input_dims, ))

    output = K.layers.Dense(hidden_layers[0], activation='relu')(input_encoder)
    for i in range(1, len(hidden_layers)):
        output = K.layers.Dense(hidden_layers[i], activation='relu')(output)
    reg = K.regularizers.l1(lambtha)
    out_encoder = K.layers.Dense(latent_dims,
                                 activation='relu',
                                 activity_regularizer=reg)(output)

    input_decoder = K.Input(shape=(latent_dims, ))
    output2 = K.layers.Dense(hidden_layers[-1],
                             activation='relu')(input_decoder)
    for i in range(len(hidden_layers)-2, -1, -1):
        output2 = K.layers.Dense(hidden_layers[i], activation='relu')(output2)
    out_decoder = K.layers.Dense(input_dims, activation='sigmoid',)(output2)

    encoder = K.models.Model(inputs=input_encoder, outputs=out_encoder)
    decoder = K.models.Model(inputs=input_decoder, outputs=out_decoder)

    input_auto = K.Input(shape=(input_dims, ))
    encoderOut = encoder(input_auto)
    decoderOut = decoder(encoderOut)
    auto = K.models.Model(inputs=input_auto, outputs=decoderOut)
    auto.compile(optimizer='Adam',
                 loss='binary_crossentropy')

    return encoder, decoder, auto
