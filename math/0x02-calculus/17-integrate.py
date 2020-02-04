#!/usr/bin/env python3

"""calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """function that calculates the integral of a polynomial"""
    if not isinstance(poly, list):
        print('entro')
        return None

    if len(poly) == 0:
        return None

    if not isinstance(C, int) and not isinstance(C, float):
        print('HOla')
        return None

    integral = [C]
    for num in range(len(poly)):
        if isinstance(poly[num], int) or isinstance(poly[num], float):
            integ = poly[num] / (num + 1)
            module = poly[num] % (num + 1)
            if module == 0:
                integ = int(integ)
            integral.append(integ)
        else:
            return None
    return integral
