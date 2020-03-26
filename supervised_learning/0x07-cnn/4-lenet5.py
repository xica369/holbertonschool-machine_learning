#!/usr/bin/python3

"""function that builds a modified version of the LeNet-5 architecture with tf

x is a tf.placeholder of shape (m, 28, 28, 1) containing the input images
for the network
  m is the number of images
  y is a tf.placeholder of shape (m, 10) with the one-hot labels to the network
The model should consist of the following layers in order:
  Convolutional layer with 6 kernels of shape 5x5 with same padding
  Max pooling layer with kernels of shape 2x2 with 2x2 strides
  Convolutional layer with 16 kernels of shape 5x5 with valid padding
  Max pooling layer with kernels of shape 2x2 with 2x2 strides
  Fully connected layer with 120 nodes
  Fully connected layer with 84 nodes
  Fully connected softmax output layer with 10 nodes

Returns:
a tensor for the softmax activated output
a training operation that utilizes Adam optimization
(with default hyperparameters)
a tensor for the loss of the netowrk
a tensor for the accuracy of the network"""

import tensorflow as tf


def lenet5(x, y):
    """LeNet-5 (Tensorflow)"""

    he_normal = tf.contrib.layers.variance_scaling_initializer()

    layer_conv = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        kernel_initializer=he_normal,
        activation=tf.nn.relu,
        padding="same")(x)

    layer = tf.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2)(layer_conv)

    layer_conv1 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        kernel_initializer=he_normal,
        activation=tf.nn.relu,
        padding="valid")(layer)

    layer1 = tf.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2)(layer_conv1)

    flact = tf.layers.Flatten()(layer1)

    fc = tf.layers.Dense(
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=he_normal)(flact)

    fc1 = tf.layers.Dense(
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=he_normal)(fc)

    fc_output = tf.layers.Dense(
        units=10,
        kernel_initializer=he_normal)(fc1)

    softmax = tf.nn.softmax(fc_output)
    loss = tf.losses.softmax_cross_entropy(y, fc_output)
    train = tf.train.AdamOptimizer().minimize(loss)

    equality = tf.equal(tf.argmax(y, axis=1), tf.argmax(fc, axis=1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return softmax, train, loss, accuracy
