#!/usr/bin/env python3

"""performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """function that performs matrix multiplication"""
    if len(mat2) == len(mat1[0]):
        new_matrix = []
        for iter in range(len(mat1)):
            temp_list = []
            for iter1 in range(len(mat2[0])):
                add = 0
                for iter2 in range(len(mat2)):
                    add = add + mat1[iter][iter2] * mat2[iter2][iter1]
                temp_list.append(add)
            new_matrix.append(temp_list)
        return new_matrix
    else:
        return None
