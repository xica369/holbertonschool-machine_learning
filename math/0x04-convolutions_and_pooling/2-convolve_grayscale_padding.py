#!/usr/bin/env python3

"""funct. that performs a convolution on grayscale images with custom padding:
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

    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    p_h = padding[0]
    p_w = padding[1]

    output_h = h + (2 * p_h) - kh + 1
    output_w = w + (2 * p_w) - kw + 1

    output = np.zeros((m, output_h, output_w))

    image = np.arange(m)
    imag_p = np.pad(images, ((0, 0), (p_h, p_h), (p_w, p_w)),
                    mode="constant", constant_values=0)

    for height in range(output_h):
        for width in range(output_w):
            output[image, height, width] = (np.sum(imag_p[image,
                                            height:height+kh,
                                            width:width+kw] *
                                            kernel, axis=(1, 2)))

    return output
