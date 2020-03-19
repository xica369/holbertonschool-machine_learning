#!/usr/bin/env python3

"""function that performs a valid convolution on grayscale images:

images: numpy.ndarray with shape (m, h, w) containing multiple grayscale images
m: the number of images
h: the height in pixels of the images
w: the width in pixels of the images
kernel: numpy.ndarray with shape (kh, kw) with the kernel for the convolution
kh: the height of the kernel
kw: the width of the kernel
Returns: a numpy.ndarray containing the convolved images"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Valid Convolution"""

    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    output_h = h - kh + 1
    output_w = w - kw + 1

    output = np.zeros((m, output_h, output_w))

    image = np.arange(m)

    for height in range(output_h):
        for width in range(output_w):
            output[image, height, width] = (np.sum(images[image,
                                            height:height+kh,
                                            width:width+kw] *
                                            kernel, axis=(1, 2)))

    return output
