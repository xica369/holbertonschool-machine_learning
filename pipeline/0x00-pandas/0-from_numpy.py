#!/usr/bin/env python3

"""
From Numpy
"""

import pandas as pd


def from_numpy(array):
    """
    Function that creates a pd.DataFrame from a np.ndarray:

    - array is the np.ndarray from which you should create the pd.DataFrame

    The columns of the pd.DataFrame should be labeled in alphabetical order
    and capitalized. There will not be more than 26 columns.

    Returns: the newly created pd.DataFrame
    """

    title_column = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    length = array.shape[1]

    df = pd.DataFrame(array)
    df.set_axis(list(title_column[:length]), axis="columns")

    return df
