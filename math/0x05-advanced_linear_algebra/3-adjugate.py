#!/usr/bin/env python3

"""calculates the adjugate matrix"""


def adjugate(matrix):
    """matrix is a list of lists whose adjugate matrix should be calculated

    Returns: the adjugate matrix of matrix"""

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if not matrix:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if len(matrix) != len(row):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix[0]) == 1:
        return [[1]]

    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = []
    for iter in range(len(cofactor_matrix)):
        temp_list = []
        for row in cofactor_matrix:
            temp_list.append(row[iter])
        adjugate_matrix.append(temp_list)
    return adjugate_matrix


def cofactor(matrix):
    """matrix is a list of lists whose cofactor matrix should be calculated

    Returns: the cofactor matrix of matrix"""

    if len(matrix[0]) == 1:
        return [[1]]

    cofactor_matrix = []
    for row in range(len(matrix)):
        minor_row = []
        for column in range(len(matrix)):
            sub_matrix = calc_sub_matrix(matrix, row, column)
            calc_determinant = determinant(sub_matrix)
            signo = (-1) ** (row + column)
            minor_row.append(calc_determinant * signo)
        cofactor_matrix.append(minor_row)

    return cofactor_matrix


def calc_sub_matrix(matrix, row, column):
    """create a new matrix by removing the given row and column"""

    new_matrix = []
    for r in range(len(matrix)):
        new_row = matrix[r][:column] + matrix[r][column + 1:]
        new_matrix.append(new_row)
    del new_matrix[row]

    return new_matrix


def determinant(matrix):
    """matrix is a list of lists whose determinant should be calculated
    The list [[]] represents a 0x0 matrix

    Returns: the determinant of matrix"""

    if matrix == [[]]:
        return 1

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

        # put zero the column elements
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
