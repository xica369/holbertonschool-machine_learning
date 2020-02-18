#!/usr/bin/env python3

"""defines a single neuron performing binary classification"""

import numpy as np
import matplotlib.pyplot as plt


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

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data.
        A is a numpy.ndarray with shape (1, m)
        containing the activated output of the neuron for each example.
        To avoid division by zero errors, use 1.0000001-A instead of 1-A
        Returns the cost"""

        y1 = 1 - Y
        y2 = 1.0000001 - A

        m = Y.shape[1]

        cost = -1 * (1 / m) * np.sum(Y * np.log(A) + y1 * np.log(y2))

        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions
        X is a numpy.ndarray with shape (nx, m) that contains the input data.
        nx is the number of input features to the neuron.
        m is the number of examples.
        Y is a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data.
        Returns the neuron’s prediction and the cost of the network
        The prediction should be a numpy.ndarray with shape (1, m)
        containing the predicted labels for each example
        The label values should be 1 if the output of the network is >= 0.5
        and 0 otherwise"""

        A = self.forward_prop(X)

        evaluate_predict = np.where(A < 0.5, 0, 1)
        cost = self.cost(Y, A)

        return (evaluate_predict, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron.
        X is a numpy.ndarray with shape (nx, m) that contains the input data.
        nx is the number of input features to the neuron.
        m is the number of examples.
        Y is a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data.
        A is a numpy.ndarray with shape (1, m)
        containing the activated output of the neuron for each example.
        alpha is the learning rate.
        Updates the private attributes __W and __b"""

        dz = A - Y
        dw = np.dot(X, dz.transpose())
        db = np.sum(dz)
        m = X.shape[1]
        self.__W = self.__W - np.dot(alpha, dw.transpose()) * (1 / m)
        self.__b = self.__b - np.dot(alpha, db) * (1 / m)

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the neuron
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Y is a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data.
        iterations is the number of iterations to train over
        alpha is the learning rate
        verbose is a boolean that defines whether or not to print information
        about the training.
        If True, print Cost after {iteration} iterations:
        {cost} every step iterations:
        Include data from the 0th and last iteration
        graph is a boolean that defines whether or not to graph information
        about the training once the training has completed. If True:
        Plot the training data every step iterations as a blue line
        Label the x-axis as iteration
        Label the y-axis as cost
        Title the plot Training Cost
        Include data from the 0th and last iteration
        Only if either verbose or graph are True:
        The 0th iteration should represent the state of the neuron before
        any training has occurred
        Returns: the evaluation of the training data after iterations
        of training have occurred
        """

        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')

        if iterations < 0:
            raise ValueError('iterations must be a positive integer')

        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')

        if alpha < 0:
            raise ValueError('alpha must be positive')

        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')

            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        costs = []
        steps = []
        for cont in range(iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if cont == iteractions or cont % step == 0:
                cost = self.cost(Y, A)

                if verbose:
                    print("Cost after {} iterations: {}".format(cont, cost))

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
