#!/usr/bin/env python3

"""Create the class TrainModel that trains a model for face verification
using triplet loss"""

from triplet_loss import TripletLoss


class TrainModel:
    """class TrainModel"""

    def __init__(self, model_path, alpha):
        """model_path is the path to the base face verification embedding model
        loads the model using with tf.keras.utils.CustomObjectScope({'tf':tf}):
        saves this model as the public instance method base_model
        alpha is the alpha to use for the triplet loss calculation
        Creates a new model:
        inputs: [A, P, N]
        A is a numpy.ndarray containing the anchor images
        P is a numpy.ndarray containing the positive images
        N is a numpy.ndarray containing the negative images
        outputs: the triplet losses of base_model
        compiles the model with Adam optimization and no additional losses
        save this model as the public instance method training_model"""

    def train(self, triplets, epochs=5, batch_size=32, validation_split=0.3,
              verbose=True):
        """instance method that trains self.training_model:
        triplets is a list containing the inputs to self.training_model
        epochs is the number of epochs to train for
        batch_size is the batch size for training
        validation_split is the validation split for training
        verbose is a boolean that sets the verbosity mode
        Returns: the History output from the training"""
