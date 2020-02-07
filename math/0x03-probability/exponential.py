#!/usr/bin/env python3

"""exponential distribution"""


class Exponential:
    """ class Exponential"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize Exponential
        data is a list of the data to be used to estimate the distribution
        lambtha is the expected number of occurences in a given time frame"""

        self.lambtha = float(lambtha)

        if lambtha < 1:
            raise ValueError("lambtha must be a positive value")

        if data is not None and not isinstance(data, list):
            raise TypeError("data must be a list")

        if isinstance(data, list) and len(data) < 2:
            raise ValueError("data must contain multiple values")

        if data is not None:
            self.lambtha = 1 / (sum(data) / len(data))

        if lambtha < 1:
            raise ValueError("lambtha must be a positive value")

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period
        x is the time period
        Returns the PDF value for x
        If x is out of range, return 0"""

        if x < 0:
            return 0

        e = 2.7182818285
        lambtha = self.lambtha

        pdf = lambtha * (e ** (-1 * lambtha * x))

        return pdf
