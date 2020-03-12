#!/usr/bin/env python3

"""Gradient Descent with L2 Regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases of a neural network using
    gradient descent with L2 regularization"""

    m = Y.shape[1]
    cp_w = weights.copy()
    dz = cache['A' + str(L)] - Y
    dw = np.dot(cache['A'+str(L-1)], dz.T) / m
    db = np.sum(dz, axis=1, keepdims=True)/m

    weights['W'+str(L)] = cp_w['W'+str(L)] - alpha*dw.T
    weights['b'+str(L)] = cp_w['b'+str(L)] - alpha*db

    for la in range(L - 1, 0, -1):
        g = cache['A'+str(la)] * (1 - cache['A'+str(la)])
        dz = np.dot(cp_w['W'+str(la+1)].T, dz) * g
        dw = np.dot(cache['A'+str(la-1)], dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        weights['W'+str(la)] = cp_w['W'+str(la)] - alpha*dw.T
        weights['b'+str(la)] = cp_w['b'+str(la)] - alpha*db
