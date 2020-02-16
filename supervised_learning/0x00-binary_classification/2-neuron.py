#!/usr/bin/env python3

"""defines a single neuron performing binary classification"""

import numpy as np


class Neuron:
    """class Neuron"""

    def __init__(self, nx):
        """nx is the number of input features to the neuron
        Public instance attributes:
        W: The weights vector for the neuron.
        b: The bias for the neuron.
        A: The activated output of the neuron (prediction).
        """

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter function to W"""
        return self.__W

    @property
    def b(self):
        """getter function to b"""
        return self.__b

    @property
    def A(self):
        """getter function to A"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        The neuron should use a sigmoid activation function
        Returns the private attribute __A"""

        W = self.__W
        b = self.__b
        z = np.dot(W, X) + b

        sigmoidea = 1 / (1 + np.exp(-1 * z))

        self.__A = sigmoidea

        return self.__A
