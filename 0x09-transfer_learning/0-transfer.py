#!/usr/bin/env python3

"""trains a convolutional neural network to classify the CIFAR 10 dataset"""

import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """function that pre-processes the data for your model:

    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
    where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

    Returns: X_p, Y_p
    X_p is a numpy.ndarray containing the preprocessed X
    Y_p is a numpy.ndarray containing the preprocessed Y"""

    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)

    return (X_p, Y_p)


if __name__ == "__main__":
    """transfer learning and model training"""

    batch_size = 50
    num_classes = 10
    epochs = 50
    model_name = "cifar10.h5"

    # the data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # convert class vectors to binary class matrices
    y_test = K.utils.to_categorical(y_test, num_classes=10)
    y_train = K.utils.to_categorical(y_train, num_classes=10)

    # data augmentation
    data_augmentation_x = np.fliplr(x_train)
    data_augmentation_y = np.fliplr(y_train)
    x_train = np.concatenate([x_train, data_augmentation_x])
    y_train = np.concatenate([y_train, data_augmentation_y])

    # preprocess
    x_train = K.applications.densenet.preprocess_input(x_train)
    x_test = K.applications.densenet.preprocess_input(x_test)
    # x_train = x_train.astype("float32") / 255
    # x_test = x_test.astype("float32") / 255

    # transfer learning with denseNet201
    pre_trained_model = K.applications.densenet.DenseNet201(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(32, 32, 3),
        pooling="avg",
        classes=num_classes)

    # set model to be trainable
    pre_trained_model.trainable = True
    set_trainable = False
    for layer in pre_trained_model.layers:
        if "conv5" in layer.name or "conv4" in layer.name:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # create a new model and add layers
    model = K.models.Sequential([
        pre_trained_model,
        K.layers.Dropout(0.3),
        K.layers.Flatten(),
        K.layers.Dense(512, activation="relu", input_shape=(32, 32, 3)),
        K.layers.Dropout(0.3),
        K.layers.Dense(256, activation="relu", input_shape=(32, 32, 3)),
        K.layers.Dropout(0.5),
        K.layers.Dense(num_classes, activation="softmax"),
    ])

    # optimizer
    opt = K.optimizers.Adam()

    # model compilation
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["acc"])

    # set callbacks
    callbacks = []

    save_best = K.callbacks.ModelCheckpoint(model_name,
                                            monitor='val_accuracy',
                                            save_best_only=True,
                                            mode="max")
    callbacks.append(save_best)
    reduce_lr = K.callbacks.ReduceLROnPlateau(monitor="val_acc",
                                              factor=.01,
                                              patience=3,
                                              min_lr=1e-5)
    callbacks.append(reduce_lr)

    # model training
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)
