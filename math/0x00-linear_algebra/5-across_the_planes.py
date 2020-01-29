#!/usr/bin/env python3

"""adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """function that adds two matrices element-wise"""
    shape_mat1 = matrix_shape(mat1)
    shape_mat2 = matrix_shape(mat2)
    if shape_mat1 == shape_mat2:
        new_matrix = []
        for row in mat1:
            row_len = len(row)
            break
        for iter in range(len(mat1)):
            for iter2 in range(row_len):
                if isinstance(mat1[iter][iter2], list):
                    add = add_arrays(mat1[iter][iter2], mat2[iter][iter2])
                    new_matrix.append(add)
                else:
                    new_matrix.append(mat1[iter][iter2] + mat2[iter][iter2])
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


def add_arrays(arr1, arr2):
    """adds two arrays element-wise"""
    if len(arr1) == len(arr2):
        add = []
        for iter in range(len(arr1)):
            add.append(arr1[iter] + arr2[iter])
        return add
    else:
        return None
