#!/usr/bin/env python3

"""Normalization Constants"""

import numpy as np
import tensorflow as tf


def normalization_constants(X):
    """"calculates the normalization (standardization) constants of a matrix:
    X is the numpy.ndarray of shape (m, nx) to normalize
    m is the number of data points
    nx is the number of features
    Returns: the mean and standard deviation of each feature"""
