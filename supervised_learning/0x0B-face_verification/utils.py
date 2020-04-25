#!/usr/bin/env python3

"""Load Images"""

import os
import glob
import cv2
import numpy as np
import csv


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


def load_csv(csv_path, params={}):
    """function that loads the contents of a csv file as a list of lists"""

    csv_list = []

    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, params)
        for row in csv_reader:
            csv_list.append(row)

    return csv_list


def save_images(path, images, filenames):
    """Function that saves images to a specific path:

    path is the path to the directory in which the images should be saved
    images is a list/numpy.ndarray of images to save
    filenames is a list of filenames of the images to save

    Returns: True on success and False on failure"""

    try:
        for iter in len(images):
            image = images[iter]
            filename = filenames[iter]
            path = "./{}/{}".format(path, filename)
            img = cv2.imread(image, 1)
            cv2.imwrite(path, img)

        return True

    except Exception:
        return False
