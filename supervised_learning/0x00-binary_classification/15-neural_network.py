#!/usr/bin/env python3

"""defines a neural network
with one hidden layer performing binary classification"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """class NeuralNetwork"""

    def __init__(self, nx, nodes):
        """nx is the number of input features
        nodes is the number of nodes found in the hidden layer
        Private instance attributes:
        W1: The weights vector for the hidden layer.
        b1: The bias for the hidden layer.
        A1: The activated output for the hidden layer.
        W2: The weights vector for the output neuron.
        b2: The bias for the output neuron.
        A2: The activated output for the output neuron (prediction)."""

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')

        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter function to W1"""
        return self.__W1

    @property
    def b1(self):
        """getter function to b1"""
        return self.__b1

    @property
    def A1(self):
        """getter function to A1"""
        return self.__A1

    @property
    def W2(self):
        """getter function to W2"""
        return self.__W2

    @property
    def b2(self):
        """getter function to b2"""
        return self.__b2

    @property
    def A2(self):
        """getter function to A2)"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Updates the private attributes __A1 and __A2
        The neurons should use a sigmoid activation function
        Returns the private attributes __A1 and __A2, respectively"""

        W1 = self.__W1
        b1 = self.__b1
        z = np.dot(W1, X) + b1

        sigmoidea = 1 / (1 + np.exp(-1 * z))
        self.__A1 = sigmoidea

        W2 = self.__W2
        b2 = self.__b2
        A1 = self.__A1
        z = np.dot(W2, A1) + b2

        sigmoidea = 1 / (1 + np.exp(-1 * z))
        self.__A2 = sigmoidea

        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data.
        A is a numpy.ndarray with shape (1, m)
        containing the activated output of the neuron for each example.
        To avoid division by zero errors, use 1.0000001 - A
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
        that contains the correct labels for the input data.
        Returns the neuron’s prediction and the cost of the network
        The prediction should be a numpy.ndarray with shape (1, m)
        containing the predicted labels for each example.
        The label values should be 1 if the output of the network is >= 0.5
        and 0 otherwise"""

        self.forward_prop(X)
        A = self.__A2

        evaluate_predict = np.where(A < 0.5, 0, 1)
        cost = self.cost(Y, A)

        return (evaluate_predict, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Y is a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data.
        A1 is the output of the hidden layer
        A2 is the predicted output
        alpha is the learning rate
        Updates the private attributes __W1, __b1, __W2, and __b2"""

        m = X.shape[1]

        dz2 = A2 - Y
        dw2 = np.dot(A1, dz2.transpose())
        db2 = np.sum(dz2, axis=1, keepdims=True)

        w2 = self.__W2
        dz1 = np.dot(w2.transpose(), dz2) * (A1 * (1 - A1))
        dw1 = np.dot(X, dz1.transpose())
        db1 = np.sum(dz1, axis=1, keepdims=True)

        self.__W2 = self.__W2 - (alpha * dw2.T) * (1 / m)
        self.__b2 = self.__b2 - (alpha * db2) * (1 / m)
        self.__W1 = self.__W1 - (alpha * dw1.T) * (1 / m)
        self.__b1 = self.__b1 - (alpha * db1) * (1 / m)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """rains the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data.
        nx is the number of input features to the neuron.
        m is the number of examples.
        Y is a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data.
        iterations is the number of iterations to train over
        alpha is the learning rate
        Updates the private attributes __W1, __b1, __A1, __W2, __b2, and __A2
        after iterations of training have occurred
        verbose is a boolean that defines whether or not to print information
        about the training. Include data from the 0th and last iteration
        graph is a boolean that defines whether or not to graph information
        about the training once the training has completed.
        Include data from the 0th and last iteration.
        Returns the evaluation of the training data after iterations of
        training have occurred
        """

        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')

        if iterations < 0:
            raise ValueError('iterations must be a positive integer')

        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')

        if alpha < 0:
            raise ValueError('alpha must be positive')

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')

            if step < 1 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        steps = []
        costs = []

        for cont in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

            if cont == iterations or cont % step == 0:
                cost = self.cost(Y, self.__A2)

                if verbose:
                    print('Cost after {} iterations: {}'.format(cont, cost))

                if graph:
                    costs.append(cost)
                    steps.append(cont)

        if graph:
            plt.plot(steps, costs)
            plt.title('Training Cost')
            plt.ylabel('cost')
            plt.xlabel('iteration')
            plt.show()

        evaluation = self.evaluate(X, Y)

        return evaluation
