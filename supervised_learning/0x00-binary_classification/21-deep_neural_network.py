#!/usr/bin/env python3
"""defines a deep neural network performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """"class DeepNeuralNetwork"""

    def __init__(self, nx, layers):
        """nx is the number of input features.
        layers is a list representing the number of nodes in each layer
        of the network.
        The first value in layers represents the number of nodes
        in the first layer.
        Sets the private instance attributes:
        __L: The number of layers in the neural network.
        __cache: A dictionary to hold all intermediary values of the network
        __weights: A dictionary to hold all weights and biased of the network.
        """

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer in range(len(layers)):
            if not isinstance(layers[layer], int) or layers[layer] <= 0:
                raise TypeError('layers must be a list of positive integers')

            if layer == 0:
                w = np.random.randn(layers[layer], nx) * np.sqrt(2 / nx)

            else:
                w = np.random.randn(layers[layer], layers[layer-1])
                w = w * np.sqrt(2 / layers[layer-1])

            self.__weights["b"+str(layer+1)] = np.zeros((layers[layer], 1))
            self.__weights["W"+str(layer+1)] = w

    @property
    def L(self):
        """getter function to L"""
        return self.__L

    @property
    def cache(self):
        """getter function to L"""
        return self.__cache

    @property
    def weights(self):
        """getter function to L"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Updates the private attribute __cache:
        The activated outputs of each layer should be saved
        in the __cache dictionary using the key A{l}
        where {l} is the hidden layer the activated output belongs to
        X should be saved to the cache dictionary using the key A0
        All neurons should use a sigmoid activation function
        Returns the output of the neural network and the cache, respectively"""

        self.__cache['A0'] = X
        for cont in range(1, self.__L + 1):
            w = self.__weights['W' + str(cont)]
            a_prev = self.__cache['A' + str(cont - 1)]
            b = self.__weights['b' + str(cont)]
            z = np.dot(w, a_prev) + b
            a = 1 / (1 + np.exp(-1 * z))
            self.__cache['A' + str(cont)] = a

        return (a, self.__cache)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data.
        A is a numpy.ndarray with shape (1, m)
        containing the activa`ted output of the neuron for each example
        To avoid division by zero errors, please use 1.0000001 - A
        Returns the cost"""

        y1 = 1 - Y
        y2 = 1.0000001 - A

        m = Y.shape[1]

        cost = -1 * (1 / m) * np.sum(Y * np.log(A) + y1 * np.log(y2))

        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Y is a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data
        Returns the neuron’s prediction and the cost of the network
        The label values should be 1 if the output of the network is >= 0.5
        and 0 otherwise"""

        self.forward_prop(X)
        A = self.__cache['A' + str(self.__L)]

        evaluate_predict = np.where(A < 0.5, 0, 1)
        cost = self.cost(Y, A)

        return (evaluate_predict, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network
        Y is a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data.
        cache is a dictionary containing all the intermediary values of
        the network.
        alpha is the learning rate
        Updates the private attribute __weights"""

        m = Y.shape[1]
        cp_w = self.__weights.copy()
        la = self.__L

        print(self.__cache)

        dz = self.__cache['A' + str(la)] - Y
        dw = np.dot(self.__cache['A' + str(la - 1)], dz.transpose())
        db = np.sum(dz, axis=1, keepdims=True)

        self.__weights['W'+str(la)] = cp_w['W'+str(la)]-(alpha*dw.T)*(1/m)
        self.__weights['b'+str(la)] = cp_w['b'+str(la)]-(alpha*db)*(1/m)

        for la in range(self.__L - 1, 0, -1):
            g = self.__cache['A'+str(la)]*(1-self.__cache['A'+str(la)])
            dz = np.dot(cp_w['W'+str(la+1)].T, dz) * g
            dw = np.dot(self.__cache['A'+str(la-1)], dz.transpose())
            db = np.sum(dz, axis=1, keepdims=True)

            self.__weights['W'+str(la)] = cp_w['W'+str(la)]-(alpha*dw.T)*(1/m)
            self.__weights['b'+str(la)] = cp_w['b'+str(la)]-(alpha*db)*(1/m)
