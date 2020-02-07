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
