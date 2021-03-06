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

    he_normal = K.initializers.he_normal()

    layer_conv1 = K.layers.Conv2D(filters=6,
                                  kernel_size=(5, 5),
                                  padding='same',
                                  kernel_initializer=he_normal,
                                  activation='relu')(X)

    layer_pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2))(layer_conv1)

    layer_conv2 = K.layers.Conv2D(filters=16,
                                  kernel_size=(5, 5),
                                  padding='valid',
                                  kernel_initializer=he_normal,
                                  activation='relu')(layer_pool1)

    layer_pool2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2))(layer_conv2)

    flatten = K.layers.Flatten()(layer_pool2)
    layer_fc1 = K.layers.Dense(units=120,
                               activation='relu',
                               kernel_initializer=he_normal)(flatten)

    layer_fc2 = K.layers.Dense(units=84,
                               activation='relu',
                               kernel_initializer=he_normal)(layer_fc1)

    softmax = K.layers.Dense(units=10,
                             activation='softmax',
                             kernel_initializer=he_normal)(layer_fc2)

    model = K.Model(inputs=X, outputs=softmax)

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
