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

        self.__W = np.random.normal(0, 0.1, 100).tolist()
        self.__b = 0
        self.__A = 0
        print('HOLA')

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
