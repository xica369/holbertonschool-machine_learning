#!/usr/bin/env python3

"""concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    new_mat = []
    if len(mat1) == len(mat2) and axis == 1:
        for iter in range(len(mat1)):
            add = mat1[iter] + mat2[iter]
            new_mat.append(add)
        return new_mat
