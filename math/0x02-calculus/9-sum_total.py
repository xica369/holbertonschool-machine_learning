#!/usr/bin/env python3

"""calculates the sum of the first n square numbers"""


def summation_i_squared(n):
    """function that calculates the sum of the first n square numbers"""

    if not isinstance(n, int):
        return None

    if n < 1:
        return None

    sum = int((n / 6) * (n + 1) * (2 * n + 1))
    return sum
