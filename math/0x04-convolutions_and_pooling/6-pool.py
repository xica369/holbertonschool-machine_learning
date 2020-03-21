#!/usr/bin/env python3

"""function that performs pooling on images:

images: is a numpy.ndarray with shape (m, h, w, c) containing multiple images
  m: is the number of images
  h: is the height in pixels of the images
  w: is the width in pixels of the images
  c: is the number of channels in the image
kernel_shape: is a tuple of (kh, kw) with the kernel shape for the pooling
  kh: is the height of the kernel
  kw: is the width of the kernel
stride: is a tuple of (sh, sw)
  sh: is the stride for the height of the image
  sw: is the stride for the width of the image
mode indicates the type of pooling
  max indicates max pooling
  avg indicates average pooling

Returns: a numpy.ndarray containing the pooled images"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Pooling """

    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    sh = stride[0]
    sw = stride[1]

    if isinstance(padding, tuple):
        p_h = padding[0]
        p_w = padding[1]

    if padding == "same":
        p_h = int(((h-1)*sh+kh-h)/2) + 1
        p_w = int(((w-1)*sw+kw-w)/2) + 1

    if padding == "valid":
        p_h = 0
        p_w = 0

    i_p = np.pad(images, pad_width=((0, 0), (p_h, p_h), (p_w, p_w)),
                 mode='constant', constant_values=0)
    output_h = int(((h + (2 * p_h) - kh) / sh) + 1)
    output_w = int(((w + (2 * p_w) - kw) / sw) + 1)

    output = np.zeros((m, output_h, output_w))
    img = np.arange(m)
    for height in range(output_h):
        for width in range(output_w):
            _h = (height*sh)+kh
            _w = (width*sw)+kw
            output[img, height, width] = (np.sum(i_p[img,
                                                     height*sh:_h,
                                                     width*sw:_w] *
                                                 kernel, axis=(1, 2)))
    return output
