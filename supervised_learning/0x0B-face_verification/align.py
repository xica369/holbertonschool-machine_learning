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
