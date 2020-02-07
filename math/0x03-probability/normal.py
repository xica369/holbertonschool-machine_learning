#!/usr/bin/env python3

"""normal distribution"""


class Normal:
    """class Normal"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """data is a list of the data to be used to estimate the distribution
        mean is the mean of the distribution
        stddev is the standard deviation of the distribution
        Sets the instance attributes mean and stddev"""

        self.mean = float(mean)
        self.stddev = float(stddev)

        if stddev < 1:
            raise ValueError("stddev must be a positive value")

        if data is not None and not isinstance(data, list):
            raise TypeError("data must be a list")

        if isinstance(data, list) and len(data) < 2:
            raise ValueError("data must contain multiple values")

        if data is not None:
            self.mean = sum(data) / len(data)

            sumat = 0
            for i in data:
                sumat = sumat + (i - self.mean) ** 2

            temp = sumat / len(data)

            self.stddev = temp ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value
        x is the x-value
        Returns the z-score of x"""

        mean = self.mean
        stddev = self.stddev

        z = (x - mean) / stddev

        return z

    def x_value(self, z):
        """Calculates the x-value of a given z-score
        z is the z-score
        Returns the x-value of z"""

        mean = self.mean
        stddev = self.stddev

        x = (z * stddev) + mean

        return x
