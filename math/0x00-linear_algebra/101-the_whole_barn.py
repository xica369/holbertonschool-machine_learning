#!/usr/bin/env python3

"""function that adds two matrices"""


def add_matrices(mat1, mat2):
    """function that adds two matrices"""
    shape1 = []
    while(mat1):
        if isinstance(mat1, list):
            shape1.append(len(mat1))
        if isinstance(mat1[0], list):
            mat1 = mat1[0]
        else:
            mat1 = None

    shape2 = []
    while(mat2):
        if isinstance(mat2, list):
            shape2.append(len(mat2))
        if isinstance(mat2[0], list):
            mat2 = mat2[0]
        else:
            mat2 = None

    if shape1 != shape2:
        return None

    else:
        return [x + y for x, y in zip(mat1, mat2)]
