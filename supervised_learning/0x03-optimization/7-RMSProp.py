#!/usr/bin/env python3

"""RMSProp"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a variable using the RMSProp optimization algorithm:
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    s is the previous second moment of var
    Returns: the updated variable and the new moment, respectively
    """

    e = epsilon
    v = beta2 * s + (1 - beta2) * pow(grad, 2)
    w = var - (grad * alpha / pow(v + e, 0.5))

    return (w, v)
