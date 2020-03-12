#!/usr/bin/env python3

"""Gradient Descent with Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates the weights of a neural network with Dropout regularization
    using gradient descent"""
