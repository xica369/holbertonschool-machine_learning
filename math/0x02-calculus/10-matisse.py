#!/usr/bin/env python3

"""calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """function that calculates the derivative of a polynomial"""
    derivate = []
    for iter in range(1, len(poly)):
        num = poly[iter]
        if isinstance(num, int):
            deriv = num * iter
            derivate.append(deriv)
        else:
            return None
    return derivate
