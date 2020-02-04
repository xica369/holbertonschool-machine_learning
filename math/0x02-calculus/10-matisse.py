#!/usr/bin/env python3

"""calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """function that calculates the derivative of a polynomial"""
    if not isinstance(poly, list):
        return None

    derivate = []
    len_poly = len(poly)

    if len_poly == 0:
        return None
    if len_poly == 1:
        return [0]
    for iter in range(1, len_poly):
        num = poly[iter]
        if isinstance(num, int) or isinstance(num, float):
            deriv = num * iter
            derivate.append(deriv)
        else:
            return None
    return derivate
