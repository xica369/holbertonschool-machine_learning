#!/usr/bin/env python3

"""Save and Load Model"""

import tensorflow.keras as K


def save_model(network, filename):
    """saves an entire model:
    network is the model to save
    filename is the path of the file that the model should be saved to
    Returns: None"""

    network.save(filename)
    return None


def load_model(filename):
    """loads an entire model:
    filename is the path of the file that the model should be loaded from
    Returns: the loaded model"""

    load_model = K.models.load_model(filename)

    return load_model
