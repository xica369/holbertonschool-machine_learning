#!/usr/bin/env python3
"""poisson distribution"""


class Poisson:
    """class Poisson"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson"""

        self.lambtha = float(lambtha)

        if lambtha < 1:
            raise ValueError("lambtha must be a positive value")

        if data is not None and not isinstance(data, list):
            raise TypeError("data must be a list")

        if isinstance(data, list) and len(data) < 2:
            raise ValueError("data must contain multiple values")

        if data is not None:
            self.lambtha = sum(data) / len(data)

        if lambtha < 1:
            raise ValueError("lambtha must be a positive value")

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”
        k is the number of “successes”"""
        if not isinstance(k, int):
            k = int(k)

        if k < 1:
            return 0

        num = (self.lambtha) ** k
        e = 2.7182818285
        fact = 1

        for n in range(1, k + 1):
            fact = fact * n

        pmf = (num / fact) * (e ** -self.lambtha)

        return pmf
