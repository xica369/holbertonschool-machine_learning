#!/usr/bin/env python3

"""Train with mini-batch gradient decet"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""

    def scheduler(epoch):
        """calcule learning rate schedule"""
        learning_rate = alpha / (1 + decay_rate * epoch)
        return learning_rate

    callbacks = []

    if validation_data:
        decay = K.callbacks.LearningRateScheduler(scheduler)
        callbacks.append(decay)

    if validation_data and early_stopping:
        early_stop = K.callbacks.EarlyStopping(patience=patience)
        callbacks.append(early_stop)

    else:
        callbacks = None

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)

    return history
