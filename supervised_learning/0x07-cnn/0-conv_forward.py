#!/usr/bin/env python3

"""function that performs forward propagation over a convolutional layer of
a neural network:

A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
containing the output of the previous layer
  m is the number of examples
  h_prev is the height of the previous layer
  w_prev is the width of the previous layer
  c_prev is the number of channels in the previous layer
W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
the kernels for the convolution
  kh is the filter height
  kw is the filter width
  c_prev is the number of channels in the previous layer
  c_new is the number of channels in the output
b is a numpy.ndarray of shape (1, 1, 1, c_new) containing
the biases applied to the convolution
activation is an activation function applied to the convolution
padding is a string that is either same or valid, indicating the type padding
stride is a tuple of (sh, sw) containing the strides for the convolution
  sh is the stride for the height
  sw is the stride for the width

Returns: the output of the convolutional layer"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Convolutional Forward Prop"""

    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = W.shape[0]
    kw = W.shape[1]
    c_prev = W.shape[2]
    c_new = W.shape[3]

    sh = stride[0]
    sw = stride[1]

    if padding == "valid":
        ph = 0
        pw = 0

    if padding == "same":
        ph = int(np.ceil((h_prev * sh - sh + kh - h_prev) / 2))
        pw = int(np.ceil((w_prev * sw - sw + kw - w_prev) / 2))

    output_h = int((h_prev + 2 * ph - kh) / sh) + 1
    output_w = int((w_prev + 2 * pw - kw) / sw) + 1
    output = np.zeros((m, output_h, output_w, c_new))

    image = np.arange(m)
    img_pad = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode="constant", constant_values=0)

    for height in range(output_h):
        for width in range(output_w):
            for channel in range(c_new):
                _h = height * sh + kh
                _w = width * sw + kw

                weight = W[:, :, :, channel]
                bias = b[:, :, :, channel]
                a = img_pad[image,
                            height*sh: _h,
                            width*sw: _w]
                z = np.sum((a * weight), axis=(1, 2, 3)) + bias

                output[image, height, width, channel] = activation(z)

    return output
