#!/usr/bin/env python3

"""
Calculates the probability density function of a Gaussian distribution:

X is a numpy.ndarray of shape (n, d) containing the data points whose PDF
should be evaluated
m is a numpy.ndarray of shape (d,) containing the mean of the distribution
S is a numpy.ndarray of shape (d, d) with the covariance of the distribution

Returns: P, or None on failure
P is a numpy.ndarray of shape (n,) with the PDF values for each data point
All values in P should have a minimum value of 1e-300
"""

import numpy as np


def pdf(X, m, S):
    """calculates pdf"""

    try:
        if X.ndim != 2 or m.ndim != 1 or S.ndim != 2:
            return None

        d = X.shape[1]

        if m.shape != (d,) or S.shape != (d, d):
            return None

        if X.shape[0] < 1 or X.shape[1] < 1:
            return None

        x_m = X - m
        cov_inv = np.linalg.inv(S)
        cov_det = np.linalg.det(S)

        den = np.sqrt(pow(2 * np.pi, d) * cov_det)
        fac = np.einsum('...k,kl,...l->...', x_m, cov_inv, x_m)
        exp = np.exp(fac * -0.5)
        pdf = (1 / den) * exp

        pdf = np.maximum(pdf, 1e-300)

    except Exception:
        return None

    return pdf
