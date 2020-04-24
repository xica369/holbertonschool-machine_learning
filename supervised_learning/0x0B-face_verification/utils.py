#!/usr/bin/env python3

"""Load Images"""

import os
import glob
import cv2
import numpy as np


def load_images(images_path, as_array=True):
    """function loads images from a directory or file"""

    images = []
    filenames = []
    image_paths = glob.glob(images_path + "/*")

    for image in sorted(image_paths):
        img_read = cv2.imread(image)
        img_rgb = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
        filename = image.split("/")[-1].strip()
        filenames.append(filename)

    if as_array is True:
        images = np.stack(images)

    return (images, filenames)
