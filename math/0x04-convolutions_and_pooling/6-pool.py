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
    cn = images.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    output_h = int((h - kh) / sh) + 1
    output_w = int((w - kw) / sw) + 1

    image = np.arange(m)
    output = np.zeros((m, output_h, output_w, cn))

    for height in range(output_h):
        for width in range(output_w):
            _h = (height * sh) + kh
            _w = (width * sw) + kw

            matrix = images[image,
                            height*sh:_h,
                            width*sw:_w]

            if mode == "max":
                output[image, height, width] = np.max(
                    matrix, axis=(1, 2))

            if mode == "avg":
                output[image, height, width] = np.average(
                    matrix, axis=(1, 2))

    return output
