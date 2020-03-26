#!/usr/bin/env python3

"""funct that builds a modified version of the LeNet-5 architecture with keras:

X is a K.Input of shape (m, 28, 28, 1) with the input images for the network
  m is the number of images
The model should consist of the following layers in order:
  Convolutional layer with 6 kernels of shape 5x5 with same padding
  Max pooling layer with kernels of shape 2x2 with 2x2 strides
  Convolutional layer with 16 kernels of shape 5x5 with valid padding
  Max pooling layer with kernels of shape 2x2 with 2x2 strides
  Fully connected layer with 120 nodes
  Fully connected layer with 84 nodes
  Fully connected softmax output layer with 10 nodes

Returns: a K.Model compiled to use Adam optimization
(with default hyperparameters) and accuracy metrics"""

import tensorflow.keras as K


def lenet5(X):
    """LeNet-5 (Keras)"""
