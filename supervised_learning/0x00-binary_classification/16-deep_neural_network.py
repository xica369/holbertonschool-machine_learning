#!/usr/bin/env python3
"""defines a deep neural network
performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """"class DeepNeuralNetwork"""

    def __init__(self, nx, layers):
        """nx is the number of input features.
        layers is a list representing the number of nodes in
        each layer of the network.
        The first value in layers represents the number of nodes
        in the first layer.
        Sets the public instance attributes:
        L: The number of layers in the neural network.
        cache: A dictionary to hold all intermediary values of the network
        weights: A dictionary to hold all weights and biased of the network."""

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for layer in range(len(layers)):
            if not isinstance(layers[layer], int) or layers[layer] <= 0:
                raise TypeError('layers must be a list of positive integers')

            if layer == 0:
                w = np.random.randn(layers[layer], nx) * np.sqrt(2 / nx)

            else:
                w = np.random.randn(layers[layer], layers[layer-1])
                w = w * np.sqrt(2 / layers[layer-1])

            self.weights["b"+str(layer+1)] = np.zeros((layers[layer], 1))
            self.weights["W"+str(layer+1)] = w
