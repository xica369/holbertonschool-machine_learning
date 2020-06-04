#!/usr/bin/env python3

"""
Create: class GaussianProcess that represents a noiseless 1D Gaussian process
"""

import numpy as np


class GaussianProcess:
    """
    class Gaussian Process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of the
        black-box function
        """

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.Y)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two matrices:

        X1 is a numpy.ndarray of shape (m, 1)
        X2 is a numpy.ndarray of shape (n, 1)
        the kernel should use the Radial Basis Function (RBF)

        Returns: covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        sqdist = (np.sum(X1**2, 1).reshape(-1, 1) +
                  np.sum(X2**2, 1) -
                  2 * np.dot(X1, X2.T))

        covariance = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

        return covariance

    def predict(self, X_s):
        """
        predicts the mean and standard deviation of points in a
        Gaussian process:

        X_s is a numpy.ndarray of shape (s, 1) containing all of the points
        whose mean and standard deviation should be calculated
        s is the number of sample points

        Returns: mu, sigma
        mu is a numpy.ndarray of shape (s,) containing the mean for each point
        in X_s, respectively
        sigma is a numpy.ndarray of shape (s,) containing the
        standard deviation for each point in X_s, respectively
        """

        s = X_s.shape[0]

        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s) + 1e-8 * np.eye(len(X_s))
        K_inv = np.linalg.inv(self.K)

        # calculate the mean and covariance matrix
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(s,)
        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)

        # calculate standard deviation at each point
        sigma = np.diag(cov)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        updates a Gaussian Process:
        X_new is a numpy.ndarray of shape (1,) that represents the new
        sample point
        Y_new is a numpy.ndarray of shape (1,) that represents the new
        sample function value
        Updates the public instance attributes X, Y, and K
        """

        X = np.append(self.X, X_new)
        self.X = X.reshape(len(X), 1)
        Y = np.append(self.Y, Y_new)
        self.Y = Y.reshape(len(Y), 1)

        self.K = self.kernel(self.X, self.X)
