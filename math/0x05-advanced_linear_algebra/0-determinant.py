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

    if matrix == [[]]:
        return 1

    for row in matrix:
        if len(matrix) != len(row):
            raise ValueError("matrix must be a square matrix")

    if len(matrix[0]) == 1:
        return matrix[0][0]

    # bottom triangle at zero with Gauss method
    signo = 1
    for row in range(len(matrix)):
        # row change when the diagonal element is 0
        if matrix[row][row] == 0:
            copy_row = matrix[row]
            for row2 in range(row + 1, len(matrix)):
                if matrix[row2][row] != 0:
                    matrix[row] = matrix[row2]
                    matrix[row2] = copy_row
                    signo *= -1
                    break

        if matrix[row][row] == 0:
            return 0

        # poner a cero los elementos de la columna
        for row2 in range(row + 1, len(matrix)):
            temp = matrix[row2][row] / matrix[row][row]
            for colum in range(row, len(matrix)):
                pos1 = matrix[row2][colum]
                pos2 = temp * matrix[row][colum]
                matrix[row2][colum] = pos1 - pos2

    # determinant calculation
    determinant = 1
    for row in range(len(matrix)):
        determinant *= matrix[row][row]

    determinant *= signo

    return(round(determinant))
