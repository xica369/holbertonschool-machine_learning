#!/usr/bin/env python3

"""calculates the determinant of a matrix"""


def determinant(matrix):
    """matrix is a list of lists whose determinant should be calculated
    The list [[]] represents a 0x0 matrix

    Returns: the determinant of matrix"""

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if not matrix:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) != len(matrix[0]) and matrix != [[]]:
        raise ValueError("matrix must be a square matrix")

    if matrix == [[]]:
        return 1

    if len(matrix[0]) == 1 and len(matrix) == 1:
        return matrix[0][0]

    return calc_det(matrix)


def calc_det(matrix):
    """
    calculate the determinant of the given matrix
    """

    if len(matrix) > 2:
        determinant = 0
        for pos in range(len(matrix)):

            # remove first row
            sub_matrix = matrix[1:]

            # remove column from position that are calculating
            for i in range(len(sub_matrix)):
                sub_matrix[i] = sub_matrix[i][:pos] + sub_matrix[i][pos + 1:]

            sign = (-1) ** pos
            # determinante + An * det(sub matrix)
            determinant += sign * matrix[0][pos] * calc_det(sub_matrix)

        return determinant
    else:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
