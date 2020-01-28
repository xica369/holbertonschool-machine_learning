#!/usr/bin/env python3

"""calculates the shape of a matrix"""


def matrix_shape(matrix):
    """function that calculates the shape of a matrix"""
    shape = []
    while(matrix):
        if isinstance(matrix, list):
            shape.append(len(matrix))
        if isinstance(matrix[0], list):
            matrix = matrix[0]
        else:
            matrix = None
    return shape
