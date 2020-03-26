#!/usr/bin/env python3

"""function that performs forward propagation over a pooling layer
of a neural network:

A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
containing the output of the previous layer
  m is the number of examples
  h_prev is the height of the previous layer
  w_prev is the width of the previous layer
  c_prev is the number of channels in the previous layer
kernel_shape is a tuple of (kh, kw) containing the size of the kernel
for the pooling
  kh is the kernel height
  kw is the kernel width
stride is a tuple of (sh, sw) containing the strides for the pooling
  sh is the stride for the height
  sw is the stride for the width
mode is a string containing either max or avg, indicating whether to perform
maximum or average pooling, respectively

Returns: the output of the pooling layer"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Pooling Forward Prop"""

    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    output_h = int((h_prev - kh) / sh) + 1
    output_w = int((w_prev - kw) / sw) + 1

    image = np.arange(m)
    output = np.zeros((m, output_h, output_w, c_prev))

    for height in range(output_h):
        for width in range(output_w):
            _h = (height*sh)+kh
            _w = (width*sw)+kw
            matrix = A_prev[image,
                            height*sh:_h,
                            width*sw:_w]

            if mode == "max":
                output[image, height, width] = np.max(
                    matrix,
                    axis=(1, 2))

            if mode == "avg":
                output[image, height, width] = np.average(
                    matrix,
                    axis=(1, 2))

    return output
