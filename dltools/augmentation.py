import cv2
import numpy as np
from . import utility
import math


class TranslationAugmentor(object):
    """
    Augments the images by translating the content and applying reflection padding.
    """

    def __init__(self, offset=40, p=1):
        """
        Initializes a new instance of the TranslationAugmentor class.

        :param offset: The offset by which the image is randomly translated.
        :param p: The probability that this will be applied.
        """
        self.offset = offset
        self.p = p
        self.down_scale = 1

    def augment(self, data):
        """
        Augments the images by translating the content and applying reflection padding.
        :param data: An array of two elements (images, targets)
        :return:
        """
        num_examples = len(data[0])

        for i in range(num_examples):
            if np.random.uniform(0, 1) > self.p:
                continue

            # Sample an offset
            offset = [self.down_scale * np.random.randint(-self.offset, self.offset + 1) for l in range(2)]
            # Extract the image and the label
            data[0][i] = self.embed_image(data[0][i], offset)

            for j in range(1, len(data)):
                data[j][i] = self.embed_labels(data[j][i], offset)

        return data

    def embed_image(self, image, offset):
        """
        Embeds the image and performs reflection padding.

        :param image: The image to translate.
        :param offset: The offset by which we translate.
        :return: The augmented image.
        """
        # Extract the image region that is defined by the offset
        region = image[:, max(-offset[0], 0):image.shape[1] - max(0, offset[0]),
                 max(-offset[1], 0):image.shape[2] - max(0, offset[1])]

        # Pad the image using reflection padding
        padding = (
            (0, 0),
            (max(0, offset[0]), max(0, -offset[0])),
            (max(0, offset[1]), max(0, -offset[1]))
        )

        region = np.pad(region, padding, 'reflect')

        return region

    def embed_labels(self, label, offset, dont_cares=-1):
        """
        Embeds the labels in a -1 map.

        :param label: The label image.
        :param offset: The offset by which we translate the image.
        :return: The augmented label image.
        """
        new_image = dont_cares * np.ones_like(label)
        new_image[max(0, offset[0]):label.shape[0] + min(0, offset[0]),
        max(0, offset[1]):label.shape[1] + min(0, offset[1])] = label[
                                                                max(-offset[0], 0):label.shape[0] - max(0, offset[0]),
                                                                max(-offset[1], 0):label.shape[1] - max(0, offset[1])]
        return new_image


class GammaAugmentor(object):
    """
    Performs random gamma augmentation on the first entry of the data array.
    """

    def __init__(self, gamma_range=(-0.1, 0.1), p=1.0):
        """
        Initializes a new instance of the GammaAugmentor class.
        :param gamma_range: The range from which to sample gamma.
        :param p: The probability that this will be applied.
        """
        self.gamma_range = gamma_range
        self.p = p

    def augment(self, data, ):
        """
        Augments the images.

        :param data: The list of images (First axis).
        :return: Augmented data
        """
        num_images = len(data[0])

        for i in range(num_images):
            if np.random.uniform(0, 1) > self.p:
                continue

            # Convert the values to [0, 1]
            aug_image = data[0][i] / 255

            # Sample a gamma factor
            Z = np.random.uniform(self.gamma_range[0], self.gamma_range[1])

            # Apply the non-linear transformation
            gamma = math.log(0.5 + 1 / math.sqrt(2) * Z) / math.log(0.5 - 1 / math.sqrt(2) * Z)

            # Perform the gamma correction
            aug_image **= gamma

            # Convert the image back [0, 255]
            aug_image *= 255
            data[0][i] = aug_image

        return data


class CastAugmentor(object):
    """
    Converts the first entry of the data array to float and the second entry to int32.
    """

    def augment(self, data):
        """
        Performs the casting augmentation
        """
        data[0] = data[0].astype('float32')

        for i in range(1, len(data)):
            data[i] = data[i].astype('int32')

        return data


class CombinedAugmentor(object):
    """
    This augmentor applies several other augmentation steps in a sequence.
    """

    def __init__(self, augmentors=None):
        if augmentors is None:
            augmentors = [
                CastAugmentor(),
                TranslationAugmentor(offset=40),
                GammaAugmentor(gamma_range=(-0.05, 0.05))
            ]
        self.augmentors = augmentors

    def augment(self, data):
        """
        Augments the data

        Parameters
        ----------
        data The data to augment

        Returns
        -------
        The augmented data.
        """
        for s in self.augmentors:
            data = s.augment(data)

        return data

