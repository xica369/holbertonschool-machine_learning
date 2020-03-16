#!/usr/bin/env python3

"""Builds a neural network"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library:

    nx: number of input features to the network
    layers: list containing the number of nodes in each layer of the network
    activations: list with the activation functions used for each layer
    lambtha: the L2 regularization parameter
    keep_prob: the probability that a node will be kept for dropout
    Returns: the keras model"""

    inputs = K.Input(shape=(nx,))
    outputs = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha),
        name="dense")(inputs)

    for layer in range(1, len(layers)):
        dropout = K.layers.Dropout(keep_prob)(outputs)
        outputs = K.layers.Dense(
            layers[layer],
            activation=activations[layer],
            kernel_regularizer=K.regularizers.l2(lambtha),
            name="dense_" + str(layer))(dropout)
        print(layers[layer])

    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model
