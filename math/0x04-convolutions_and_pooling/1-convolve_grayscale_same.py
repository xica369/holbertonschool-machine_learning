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

    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    if kh % 2 == 0:
        p_h = int(kh / 2)
    else:
        p_h = int((kh - 1) / 2)

    if kw % 2 == 0:
        p_w = int(kw / 2)
    else:
        p_w = int((kw - 1) / 2)

    output = np.zeros((m, h, w))

    image = np.arange(m)
    img_pad = np.pad(images, pad_width=((0, 0), (p_h, p_h), (p_w, p_w)),
                     mode="constant", constant_values=0)

    for height in range(h):
        for width in range(w):
            output[image, height, width] = (np.sum(img_pad[image,
                                            height:height+kh,
                                            width:width+kw] *
                                            kernel, axis=(1, 2)))

    return output
