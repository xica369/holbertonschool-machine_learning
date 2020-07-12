#!/usr/bin/env python3

"""
Function that creates the loss tensor and training op for the generator:

Z is the tf.placeholder that is the input for the generator
X is the tf.placeholder that is the input for the discriminator

Returns: loss, train_op
loss is the generator loss
train_op is the training operation for the generator
"""

import tensorflow as tf
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_generator(Z):
    """
    Function that creates the loss tensor and training op for the generator
    """

    # generator model
    generator_sample = generator(Z)

    # discriminator model
    discriminator_fake = discriminator(generator_sample)

    # loss function
    loss = -tf.reduce_mean(tf.log(discriminator_fake))

    # select parameters
    gen_vars = []

    for variable in tf.trainable_variables():
        if variable.name.startwith("gener"):
            gen_vars.append(variable)

    # optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss, var_list=gen_vars)

    return loss, train_op
