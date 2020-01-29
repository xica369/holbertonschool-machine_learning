#!/usr/bin/env python3

"""adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """function that adds two matrices element-wise"""
    shape_mat1 = matrix_shape(mat1)
    shape_mat2 = matrix_shape(mat2)
    if shape_mat1 == shape_mat2:
        new_matrix = []
        for iter in range(len(mat1)):
            temp_list = []
            for iter2 in range(len(mat1[0])):
                add = mat1[iter][iter2] + mat2[iter][iter2]
                temp_list.append(add)
            new_matrix.append(temp_list)
        return new_matrix
    else:
        return None


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
