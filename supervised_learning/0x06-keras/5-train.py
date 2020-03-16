#!/usr/bin/env python3

"""Train with mini-batch gradient decet"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""

    history = None
    if validation_data:
        history = network.fit(x=data,
                              y=labels,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose,
                              shuffle=shuffle)

    return history
