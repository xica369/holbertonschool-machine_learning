#!/usr/bin/env python3

"""Create the class FaceAlign"""

import dlib


class FaceAlign:
    """class FaceAlign"""

    def __init__(self, shape_predictor_path):
        """shape_predictor_path is the path to the dlib shape predictor model
        Sets the public instance attributes:
        detector - contains dlib‘s default face detector
        shape_predictor - contains the dlib.shape_predictor"""

        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
