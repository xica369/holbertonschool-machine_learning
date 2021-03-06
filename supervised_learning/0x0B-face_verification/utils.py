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

    if not (os.path.exists(path)):
        return False

    for index in range(len(images)):
        filename = filenames[index]
        image = images[index]
        path_img = "./{}/{}".format(path, filename)
        img_readed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path_img, img_readed)
    return True


def generate_triplets(images, filenames, triplet_names):
    """function that generates triplets:

    images is a numpy.ndarray of shape (n, h, w, 3)
    containing the various images in the dataset
    filenames is a list of length n containing the corresponding
    filenames for images
    triplet_names is a list of lists where each sublist contains the filenames
    of an anchor, positive, and negative image, respectively

    Returns: a list [A, P, N]
    A is a numpy.ndarray of shape (m, h, w, 3)
    containing the anchor images for all m triplets
    P is a numpy.ndarray of shape (m, h, w, 3)
    containing the positive images for all m triplets
    N is a numpy.ndarray of shape (m, h, w, 3)
    containing the negative images for all m triplets"""

    indices_A = []
    indices_P = []
    indices_N = []

    # remove ".jpg"
    new_filenames = [filename.split(".")[0] for filename in filenames]

    for triplet_name in triplet_names:
        for name in triplet_name:
            if name not in new_filenames:
                # Reeplace special characters as í, é, ñ
                new_name = name.encode('utf-8').decode('utf-8')
                new_name = new_name.replace('eÌ\x81', 'é')
                new_name = new_name.replace('iÌ\x81', chr(105) + chr(769))
                new_name = new_name.replace('nÌƒ', chr(110) + chr(771))

                # replace name in triplet_names
                triplet_name[triplet_name.index(name)] = new_name

        # save image indices
        indices_A.append(new_filenames.index(triplet_name[0]))
        indices_P.append(new_filenames.index(triplet_name[1]))
        indices_N.append(new_filenames.index(triplet_name[2]))

    A = images[indices_A]
    P = images[indices_P]
    N = images[indices_N]

    return [A, P, N]
