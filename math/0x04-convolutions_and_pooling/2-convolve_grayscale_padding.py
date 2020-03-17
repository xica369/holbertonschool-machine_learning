#!/usr/bin/env python3

"""function that performs a convolution on grayscale images with custom padding:

images: numpy.ndarray with shape (m, h, w) containing multiple grayscale images
m: is the number of images
h: is the height in pixels of the images
w: is the width in pixels of the images
kernel: numpy.ndarray with shape (kh, kw) with the kernel for the convolution
kh: is the height of the kernel
kw: is the width of the kernel
padding: is a tuple of (ph, pw)
ph: is the padding for the height of the image
pw: is the padding for the width of the image

Returns: a numpy.ndarray containing the convolved images"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Convolution with Padding"""
