#!/usr/bin/env python3

"""binomial distribution"""


class Binomial:
    """class Binomial"""
    def __init__(self, data=None, n=1, p=0.5):
        """data is a list of the data to be used to estimate the distribution
        n is the number of Bernoulli trials
        p is the probability of a “success”
        Sets the instance attributes n and p"""

        self.n = n
        self.p = p

        if n < 1:
            raise ValueError("n must be a positive value")

        if p <= 0 and p >= 1:
            raise ValueError("p must be greater than 0 and less than 1")

        if data is not None and not isinstance(data, list):
            raise TypeError("data must be a list")

        if isinstance(data, list) and len(data) < 2:
            raise ValueError("data must contain multiple values")

        if data is not None:
            pi = 3.1415926536
            N = len(data)

            fact_N = 1
            fact_x = 1

            for i in range(len(data)):
                if i != 0:
                    fact_N = fact_N * i

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”
        k is the number of “successes”
        If k is not an integer, convert it to an integer
        If k is out of range, return 0
        Returns the PMF value for k"""

        if not isinstance(k, int):
            k = int(k)

        return 0
