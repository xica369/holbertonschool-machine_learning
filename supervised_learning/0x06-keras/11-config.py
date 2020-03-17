#!/usr/bin/env python3

"""Save and Load Configuration"""

import tensorflow.keras as K


def save_config(network, filename):
    """saves a model’s configuration in JSON format:
    network is the model whose configuration should be saved
    filename is the path of the file that the configuration should be saved to
    Returns: None"""

    network.to_json(filename)

    return None


def load_config(filename):
    """loads a model with a specific configuration:
    filename: path of the file with the model’s configuration in JSON format
    Returns: the loaded model"""

    model = K.models.model_from_json(filename)

    return model
