"""Utility functions for training SemSeg systems on CityScapes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np

cityscapes_color_map = {
    -1: (0, 0, 0), 
    255: (0, 0, 0),
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
}


cityscapes_value_map = list(range(256))
cityscapes_value_map[0] = 255
cityscapes_value_map[1] = 255
cityscapes_value_map[2] = 255
cityscapes_value_map[3] = 255
cityscapes_value_map[4] = 255
cityscapes_value_map[5] = 255
cityscapes_value_map[6] = 255
cityscapes_value_map[7] = 0
cityscapes_value_map[8] = 1
cityscapes_value_map[9] = 255
cityscapes_value_map[10] = 255
cityscapes_value_map[11] = 2
cityscapes_value_map[12] = 3
cityscapes_value_map[13] = 4
cityscapes_value_map[14] = 255
cityscapes_value_map[15] = 255
cityscapes_value_map[16] = 255
cityscapes_value_map[17] = 5
cityscapes_value_map[18] = 255
cityscapes_value_map[19] = 6
cityscapes_value_map[20] = 7
cityscapes_value_map[21] = 8
cityscapes_value_map[22] = 9
cityscapes_value_map[23] = 10
cityscapes_value_map[24] = 11
cityscapes_value_map[25] = 12
cityscapes_value_map[26] = 13
cityscapes_value_map[27] = 14
cityscapes_value_map[28] = 15
cityscapes_value_map[29] = 255
cityscapes_value_map[30] = 255
cityscapes_value_map[31] = 16
cityscapes_value_map[32] = 17
cityscapes_value_map[33] = 18


def create_color_label_image(np_array):
    """Converts a color image from an id images.

    Args:
        np_array: A numpy id image.

    Returns:
        An RGB image
    """
    if len(np_array.shape) == 3:
        max_mask = np.max(np_array, axis=0)
        np_array = np.argmax(np_array, axis=0)
        np_array[max_mask == 0] = -1

    result = np.zeros(np_array.shape + (3, ), dtype="uint8")

    # Convert the image
    for i in range(np_array.shape[0]):
        for j in range(np_array.shape[1]):
            result[i, j, :] = cityscapes_color_map[np_array[i, j]]

    result = result[:, :, ::-1]

    return result


def get_cityscapes_path():
    """Returns the path to the cityscapes folder."""
    filename = "cityscapes_path.txt"
    cs_path = "/"
    # Does a file already exist?
    if os.path.exists(filename):
        # Read the path
        with open(filename) as f:
            cs_path = f.read()

    # Ask the user for the actual path
    user_input = input("Enter path to CityScapes folder [%s]: " % cs_path)

    # Did the user enter something?
    if user_input != "":
        # Yes, update the file
        with open(filename) as f:
            f.write(user_input)
        cs_path = user_input

    return cs_path


def get_interactive_input(phrase, filename, default):
    """Returns the path to the cityscapes folder."""
    value = default
    # Does a file already exist?
    if os.path.exists(filename):
        # Read the path
        with open(filename, "r") as f:
            value = f.read()

    # Ask the user for the actual path
    user_input = raw_input("%s [%s]: " % (phrase, value))

    # Did the user enter something?
    if user_input != "":
        # Yes, update the file
        with open(filename, "w") as f:
            f.write(user_input)
        value = user_input

    return value


def tensor2opencv(image):
    """Converts an image from the Theano representation to the OpenCv."""
    image = np.rollaxis(image, 0, 3)
    return image[:, :, ::-1]


def opencv2tensor(image):
    """Converts an image from the opencv representation to Theano."""
    return np.rollaxis(image[:, :, ::-1], 2)


def get_image_label_pairs(cs_folder, folder):
    """Creates a list of (image, label) tuples.

    Args:
        cs_folder: The cityscapes data folder
        folder: The folder to load (train, val)

    Returns:
        A list of tuples.
    """
    image_folder = os.path.join(cs_folder, "leftImg8bit", folder)

    # List of tuples of the form (filename, cityname)
    image_names = []

    # Get all the images in the sub folders
    for city in os.listdir(image_folder):
        city_folder = os.path.join(image_folder, city)

        for fname in os.listdir(city_folder):
            if fname.endswith(".png"):
                image_names.append((os.path.join(city_folder, fname), city))

    # Make sure that the ordering of the filenames is invariant under the
    # possible directory traversals. This is important if we want to generate
    # the demo video.
    image_names = sorted(image_names, key=lambda x: x[0])

    # Restore the original return format
    image_names, _ = map(list, zip(*image_names))

    target_names = [x.replace("leftImg8bit", "gtFine", 1) for x in image_names]
    target_names = [x.replace("leftImg8bit", "gtFine_labelIds") for x in
                    target_names]

    return list(zip(image_names, target_names))
