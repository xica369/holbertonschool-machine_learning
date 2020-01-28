#!/usr/bin/env python3

"""adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """adds two arrays element-wise"""
    if len(arr1) == len(arr2):
        add = []
        for iter in range(len(arr1)):
            add.append(arr1[iter] + arr2[iter])
        return add
    else:
        return None
