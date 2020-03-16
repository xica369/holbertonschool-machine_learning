#!/usr/bin/env python3

"""Sequential"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library:

    nx: the number of input features to the network
    layers: list containing the number of nodes in each layer of the network
    activations: list with the activation functions used for each layer
    lambtha: the L2 regularization parameter
    keep_prob: the probability that a node will be kept for dropout
    Returns: the keras model"""

    model = K.Sequential()

    model.add(K.layers.Dense(
        units=layers[0],
        input_shape=(nx, ),
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha),
        name="dense"
        ))

    for layer in range(1, len(layers)):
        model.add(K.layers.Dropout(rate=(1 - keep_prob)))
        model.add(K.layers.Dense(
            units=layers[layer],
            activation=activations[layer],
            kernel_regularizer=K.regularizers.l2(lambtha),
            name="dense_" + str(layer)
        ))

    return model
