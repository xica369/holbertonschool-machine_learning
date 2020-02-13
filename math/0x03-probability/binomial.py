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

        if p <= 0 or p >= 1:
            raise ValueError("p must be greater than 0 and less than 1")

        if data is not None and not isinstance(data, list):
            raise TypeError("data must be a list")

        if isinstance(data, list) and len(data) < 2:
            raise ValueError("data must contain multiple values")

        if data is not None:
            pi = 3.1415926536
            N = len(data)

            mean = sum(data) / N

            num = 0

            for x in data:
                num = num + (x - mean) ** 2

            stddev = num / N

            self.p = 1 - (stddev / mean)

            self.n = int(mean / self.p)

            self.p = mean / self.n

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”
        k is the number of “successes”
        If k is not an integer, convert it to an integer
        If k is out of range, return 0
        Returns the PMF value for k"""

        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        p = self.p
        q = 1 - p
        n_fact = 1
        k_fact = 1
        n_k_fact = 1

        for i in range(self.n + 1):
            if i != 0:
                n_fact = n_fact * i

        for i in range(k + 1):
            if i != 0:
                k_fact = k_fact * i

        for i in range(self.n - k + 1):
            if i != 0:
                n_k_fact = n_k_fact * i

        n_k_comb = n_fact / (k_fact * n_k_fact)

        pmf = n_k_comb * (p ** k) * (q ** (self.n - k))

        return pmf

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”
        k is the number of “successes”
        If k is not an integer, convert it to an integer
        If k is out of range, return 0
        Returns the CDF value for k"""

        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        cdf = 0
        for num in range(k):
            cdf = cdf + self.pmf(num)

        return cdf
