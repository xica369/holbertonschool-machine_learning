#!/usr/bin/env python3

"""function that performs a convolution on images using multiple kernels:

images: is a numpy.ndarray with shape (m, h, w, c) containing multiple images
m: is the number of images
h: is the height in pixels of the images
w: is the width in pixels of the images
c: is the number of channels in the image
kernels: numpy.ndarray of shape (kh, kw, c, nc) with the kernels to convolution
  kh: is the height of a kernel
  kw: is the width of a kernel
  nc: is the number of kernels
padding: is either a tuple of (ph, pw), ‘same’, or ‘valid’
if ‘same’, performs a same convolution
if ‘valid’, performs a valid convolution
if a tuple:
  ph: is the padding for the height of the image
  pw: is the padding for the width of the image
stride: is a tuple of (sh, sw)
  sh: is the stride for the height of the image
  sw: is the stride for the width of the image

Returns: a numpy.ndarray with the convolved images"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Multiple Kernels"""

    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernels.shape[0]
    kw = kernels.shape[1]
    nc = kernels.shape[3]

    sh = stride[0]
    sw = stride[1]

    if isinstance(padding, tuple):
        p_h = padding[0]
        p_w = padding[1]

    if padding == "same":
        p_h = int(((h - 1) * sh + kh - h) / 2) + 1
        p_w = int(((w - 1) * sw + kw - w) / 2) + 1

    if padding == "valid":
        p_h = 0
        p_w = 0

    output_h = int((h + (2 * p_h) - kh) / sh) + 1
    output_w = int((w + (2 * p_w) - kw) / sw) + 1

    output = np.zeros((m, output_h, output_w, nc))

    img = np.arange(m)
    i_p = np.pad(images, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)),
                 mode="constant", constant_values=0)

    for h in range(output_h):
        for w in range(output_w):
            for cha in range(nc):
                _h = (h * sh) + kh
                _w = (w * sw) + kw
                output[img, h, w, cha] = (np.sum(i_p[img,
                                                     h * sh:_h,
                                                     w * sw:_w] *
                                                 kernels[:, :, :, cha],
                                                 axis=(1, 2, 3)))

    return output
