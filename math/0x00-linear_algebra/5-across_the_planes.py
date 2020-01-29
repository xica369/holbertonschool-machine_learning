#!/usr/bin/env python3

"""adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """function that adds two matrices element-wise"""

    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
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
