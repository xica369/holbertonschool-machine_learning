#!/usr/bin/env python3

"""returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """returns the transpose of a 2D matrix"""
    transpose = []
    for iter in range(len(matrix[0])):
        temp_list = []
        for row in matrix:
            temp_list.append(row[iter])
        transpose.append(temp_list)
    return transpose
