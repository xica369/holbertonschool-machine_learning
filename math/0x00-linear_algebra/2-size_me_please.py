#!/usr/bin/env python3

def matrix_shape(matrix):
    """function that calculates the shape of a matrix"""
    shape = []
    for row in matrix:
        shape.append(len(matrix))
        shape.append(len(row))
        if isinstance(row[0], list):
            shape.append(len(row[0]))
        break
    return shape
