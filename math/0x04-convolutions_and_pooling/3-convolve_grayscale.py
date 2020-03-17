#!/usr/bin/env python3

"""function that performs a convolution on grayscale images:

images: numpy.ndarray with shape (m, h, w) containing multiple grayscale images
m: is the number of images
h: is the height in pixels of the images
w: is the width in pixels of the images
kernel: numpy.ndarray with shape (kh, kw) with the kernel for the convolution
kh: is the height of the kernel
kw: is the width of the kernel
padding: is either a tuple of (ph, pw), ‘same’, or ‘valid’
if ‘same’, performs a same convolution
if ‘valid’, performs a valid convolution
if a tuple:
  ph is the padding for the height of the image
  pw is the padding for the width of the image
stride: is a tuple of (sh, sw)
sh: is the stride for the height of the image
sw: is the stride for the width of the image

Returns: a numpy.ndarray containing the convolved images"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Strided Convolution"""
