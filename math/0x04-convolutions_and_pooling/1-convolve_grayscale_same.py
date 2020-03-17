#!/usr/bin/env python3

"""function that performs a same convolution on grayscale images:

images: numpy.ndarray with shape (m, h, w) containing multiple grayscale images
m: the number of images
h: the height in pixels of the images
w: the width in pixels of the images
kernel: numpy.ndarray with shape (kh, kw) with the kernel for the convolution
kh: the height of the kernel
kw: the width of the kernel

Returns: a numpy.ndarray containing the convolved images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Same Convolution"""
