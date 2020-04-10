#!/usr/bin/env python3

"""trains a convolutional neural network to classify the CIFAR 10 dataset"""

import tensorflow.keras as K


def preprocess_data(X, Y):
    """function that pre-processes the data for your model:

    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
    where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

    Returns: X_p, Y_p
    X_p is a numpy.ndarray containing the preprocessed X
    Y_p is a numpy.ndarray containing the preprocessed Y"""

    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)

    return (X_p, Y_p)


if __name__ == "__main__":

    batch_size = 64
    num_classes = 10
    epochs = 3
    num_predictions = 20
    model_name = "cifar10.h5"

    # the data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # convert class vectors to binary class matrices
    y_test = K.utils.to_categorical(y_test, num_classes=10)
    y_train = K.utils.to_categorical(y_train, num_classes=10)

    # preprocess
    x_train = K.applications.vgg16.preprocess_input(x_train)
    x_test = K.applications.vgg16.preprocess_input(x_test)

    # transfer learning with vgg16
    pre_trained_model = K.applications.vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(32, 32, 3),
        pooling=None,
        classes=num_classes)

    pre_trained_model.summary()

    pre_trained_model.trainable = True
    set_trainable = False

    for layer in pre_trained_model.layers:
        if layer.name == "block5_conv1":
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model = K.models.Sequential([
        pre_trained_model,
        K.layers.Dropout(0.5),
        K.layers.Flatten(),
        K.layers.Dense(256, activation="relu", input_shape=(32, 32, 3)),
        K.layers.Dropout(0.2),
        K.layers.Dense(10, activation="sigmoid"),
    ])

    model.compile(loss="binary_crossentropy",
                  optimizer=K.optimizers.RMSprop(lr=1e-4),
                  metrics=["acc"])

    early_stop = [K.callbacks.EarlyStopping()]

    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        shuffle=False,
                        validation_data=(x_test, y_test),
                        callbacks=early_stop)

    evaluate = model.evaluate(x_test, y_test)
    print("test_loss - test_acc ", evaluate)

    model.save(model_name)
