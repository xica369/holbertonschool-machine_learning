#!/usr/bin/env python3

"""
class MultiNormal that represents a Multivariate Normal distribution:

"""

import numpy as np


class MultiNormal:
    """Create the class MultiNormal"""

    def __init__(self, data):
        """
        data is a numpy.ndarray of shape (d, n) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point

        Set the public instance variables:
        mean: numpy.ndarray of shape (d, 1) with the mean of data
        cov: numpy.ndarray of shape (d, d) with the covariance matrix data
        """

        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        if data.shape[1] < 2:
            raise TypeError("data must contain multiple data points")

        # Set the public instance variables

        d = data.shape[0]
        n = data.shape[1]
        self.d = d

        mean = np.mean(data.T, axis=0).reshape(1, d)
        self.mean = mean.T
        x = data.T - mean
        cov = np.dot(x.T, x) / (n - 1)
        self.cov = cov.T

    def pdf(self, x):
        """
        instance method that calculates the PDF at a data point:
        x: np.ndarray of shape (d, 1) with the data point to calculate PDF
        d is the number of dimensions of the Multinomial instance

        Returns the value of the PDF
        """

        if not isinstance(x, np.ndarray):
            raise TypeError("x must by a numpy.ndarray")

        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != self.d:
            raise ValueError("x mush have the shape ({}, 1)".format(self.d))

        mean = self.mean
        cov = self.cov

        part1 = 1/(((2*np.pi) ** (len(mean)/2)) * (np.linalg.det(cov)**(1/2)))
        part2 = (-1/2) * ((x-mean).T.dot(np.linalg.inv(cov))).dot((x-mean))

        pdf = float(part1 * np.exp(part2))

        return pdf
