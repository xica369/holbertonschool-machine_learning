#!/usr/bin/env python3

"""
Function that creates the loss tensor and training op for the discriminator:

Z is the tf.placeholder that is the input for the generator
X is the tf.placeholder that is the real input for the discriminator

Returns: loss, train_op
  loss is the discriminator loss
  train_op is the training operation for the discriminator
"""

import tensorflow as tf
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_discriminator(Z, X):
    """
    Function that creates the loss tensor and training op for the discriminator
    """

    # generator model
    generator_sample = generator(Z)

    # discriminator model
    discriminator_real = discriminator(X)
    discriminator_fake = discriminotor(generator_sample)

    # loss function
    loss = -tf.reduce_mean(tf.log(discriminator_real) +
                           tf.log(1 - discriminator_fake))

    # select parameters
    disc_vars = []

    for variable in tf.trainable_variables():
        if variable.name.startwith("discr"):
            disc_vars.append(variable)

    # optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss, var_list=disc_vars)

    return loss, train_op
