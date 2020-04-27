#!/usr/bin/env python3

"""Create the class FaceAlign"""

import dlib
import numpy as np
import cv2


class FaceAlign:
    """class FaceAlign"""

    def __init__(self, shape_predictor_path):
        """shape_predictor_path is the path to the dlib shape predictor model
        Sets the public instance attributes:
        detector - contains dlib‘s default face detector
        shape_predictor - contains the dlib.shape_predictor"""

        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """method that detects a face in an image"""

        try:
            detected_faces = self.detector(image, 1)

            if len(detected_faces) == 0:
                x, y, w, h = 0, 0, image.shape[1], image.shape[0]
                box_face = dlib.rectangle(x, y, w, h)

            else:
                max_area = 0
                for fc in detected_faces:
                    _x = fc.left()
                    _y = fc.top()
                    _w = fc.width()
                    _h = fc.height()

                    if _w * _h > max_area:
                        x, y, w, h = _x, _y, _w, _h
                        max_area = w * h

                box_face = dlib.rectangle(x, y, (w + x), (h + y))

            return box_face

        except Exception:
            return None

    def find_landmarks(self, image, detection):
        """that finds facial landmarks"""

        try:
            landmarks = self.shape_predictor(image, detection)
            landmarks_list = []
            num_landmarks = landmarks.num_parts
            landmarks_coordinates = np.empty((num_landmarks, 2))

            for iter in range(0, num_landmarks):
                landmarks_coordinates[iter] = [landmarks.part(iter).x,
                                               landmarks.part(iter).y]

            return landmarks_coordinates

        except Exception:
            return None

    def align(self, image, landmark_indices, anchor_points, size=96):
        """method that aligns an image for face verification"""

        detection = self.detect(image)
        landmark_coordinates = self.find_landmarks(image, detection)

        np_landmark_coordinates = np.float32(landmark_coordinates)
        initial_points = np_landmark_coordinates[landmark_indices]
        destination_points = size * anchor_points
        map_matrix = cv2.getAffineTransform(initial_points, destination_points)
        affine_dim = (size, size)

        affine_transformation = cv2.warpAffine(src=image,
                                               M=map_matrix,
                                               dsize=affine_dim,
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=0)

        return affine_transformation
