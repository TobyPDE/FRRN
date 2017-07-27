"""Contains classes for loading and augmenting data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import division

import cv2
import abc
import numpy as np
import six
import multiprocessing
import logging


cityscapes_value_map = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
                        7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4,
                        14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7,
                        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                        28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18}


class DataLoader(object):
    """Loads data from disk and applies augmentation."""

    def __init__(self, uri_queue, augmentor, image_loader, target_loader):
        """Initializes a new instance of the DataLoader class.

        Args:
            uri_queue: A queue that contains the data uris.
            augmentor: An instance of `Augmentor`.
            image_loader: An instance of `Loader` for loading the image.
            target_loader: An instance of `Loader` for loading the labels.
        """
        self._uri_queue = uri_queue
        self._augmentor = augmentor
        self._image_loader = image_loader
        self._target_loader = target_loader

    def get_pair(self):
        """Loads a new image annotation pair.

        Returns:
        A tuple consisting of the image and its annotation.
        """
        image_file, target_file = self._uri_queue.get()
        image = self._image_loader.load(image_file)
        target = self._target_loader.load(target_file)

        if self._augmentor is not None:
            image, target = self._augmentor.augment(image, target)

        return image, target


class BatchLoader(object):
    """Loads batches of data and puts them into a queue."""

    def __init__(self, data_loader, queue, batch_size, num_classes):
        """Initializes a new instance of the BatchLoader class.

        Args:
            data_loader: The underlying data loader.
            queue: The queue to push examples into.
            batch_size: The batch size.
        """
        self._data_loader = data_loader
        self._queue = queue
        self._batch_size = batch_size
        self._num_classes = num_classes
        self.start_loading()

    def start_loading(self):
        """Starts loading images in an infinite loop."""
        # Choose a new numpy random seed. Otherwise all processes use the same
        # seed.
        np.random.seed()
        while True:
            images, targets = self.load_batch()
            self._queue.put((images, targets))

    def load_batch(self):
        """Loads a single batch of images.

        Returns:
        A tuple of `images` and `targets`.
        """
        data = [self._data_loader.get_pair() for _ in range(self._batch_size)]
        # Convert the image from H, W, C and BGR to C, H, W and RGB.
        images = [self._convert_image(d[0]) for d in data]
        targets = [self._convert_target(d[1]) for d in data]

        images = np.stack(images, axis=0)
        targets = np.stack(targets, axis=0)

        return images, targets

    def _convert_image(self, image):
        """Converts the image to the desired format."""
        image = np.rollaxis(image[:, :, ::-1], 2)
        image = np.nan_to_num(image)
        return image

    def _convert_target(self, target):
        """Converts the target to int32 and replaces void labels by -1.

        Args:
            target: The target image.

        Returns:
            The converted target image.
        """
        target = target.astype("int32")
        target[target == 255] = self._num_classes

        flat_targets = target.flatten()
        class_matrix = np.eye(self._num_classes + 1, dtype='float32')

        # Select the one-hot row for each example.
        selects = class_matrix[flat_targets]

        # Get back a properly shaped tensor.
        new_shape = target.shape + (self._num_classes + 1, )
        new_target = np.reshape(selects, new_shape )
        new_target = new_target[:, :, :self._num_classes]
        new_target = new_target.transpose((2, 0, 1))

        return new_target


class DataProvider(object):
    """Client class for loading data asynchronously."""

    def __init__(self,
                 augmentor,
                 image_loader,
                 target_loader,
                 data_iterator,
                 batch_size,
                 num_classes,
                 threads=3,
                 prefetch_batches=10):
        """Initializes a new instance of the DataProvider class.

        Args:
            augmentor: A dataset augmentor.
            image_loader: Loader for loading the images.
            target_loader: Loader for loaing the targets.
            data_iterator: Data sequence iterator.
            batch_size: The batch size.
            num_classes: The number of classes.
            threads: The number of loader threads.
            prefetch_batches: The number of batches to prefetch.
        """
        # Create the queue for feeding the data uris.
        uri_queue = multiprocessing.Queue(maxsize=threads * prefetch_batches)

        # Fill the queue.
        p = multiprocessing.Process(
            target=data_iterator.fill_queue, args=(uri_queue, ))
        p.daemon = True
        p.start()

        # Create the data loader.
        loader = DataLoader(uri_queue, augmentor, image_loader, target_loader)

        # Create the data queue.
        self._queue = multiprocessing.Queue(maxsize=prefetch_batches)

        # Launch the loader.
        for _ in range(threads):
            args = (loader, self._queue, batch_size, num_classes)
            p = multiprocessing.Process(target=BatchLoader, args=args)
            p.daemon = True
            p.start()

        self._data_iterator = data_iterator
        self._batch_size = batch_size

    def get_num_batches(self):
        """Returns the number of batches."""
        return self._data_iterator.get_sequence_length() // self._batch_size

    def reset(self):
        """Resets the data iterator.

        This functionality is not supported in python.
        """
        logging.warning("Resetting the data provider is not supported in "
                        "Python. Please consider using the C++ chianti library "
                        "for training.")

    def next(self):
        """Returns the next batch.

        Returns:
            A tuple consisting of image and target.
        """
        return self._queue.get()


@six.add_metaclass(abc.ABCMeta)
class LoaderBase(object):
    """Instances of this class load images from a given resource identifier."""

    @abc.abstractmethod
    def load(self, uri):
        """Loads an image from a given resource identifier.

        Args:
            uri: The resource identifier.

        Returns:
        An image as numpy array.
        """


class RGBLoader(LoaderBase):
    """Loads an RGB image from the local disk."""

    def load(self, uri):
        """Loads an image from a given resource identifier.

        Args:
            uri: The resource identifier.

        Returns:
        An image as numpy array.
        """
        img = cv2.imread(uri, 1).astype('float32')
        img = img / 255.0
        return img


class ValueMapperLoader(LoaderBase):
    """Loads a gray scale image from disk and applies a value mapping."""

    def __init__(self, intensity_map):
        """Initializes a new instance of the ValueMapperLoader class.

        Args:
            intensity_map: The intensity map.
        """
        super(ValueMapperLoader, self).__init__()
        self._intensity_map = intensity_map
        self._map_func = np.vectorize(lambda px: self._intensity_map[px])

    def load(self, uri):
        """Loads an image from a given resource identifier.

        Args:
            uri: The resource identifier.

        Returns:
        An image as numpy array.
        """
        img = cv2.imread(uri, 0)
        img = self._map_func(img)
        return img


@six.add_metaclass(abc.ABCMeta)
class IteratorBase(object):
    """Allows to iterate over sequences in different orders."""

    def __init__(self, sequence):
        """Initializes a new instance of the IteratorBase class.

        Args:
            sequence: The sequence to iterate over.
        """
        self._mutex = multiprocessing.Lock()
        self._sequence = sequence

        if not self._sequence:
            raise ValueError("Empty iteration sequence.")

    def fill_queue(self, queue):
        """Fills the queue with the data from the iterator."""
        while True:
            queue.put(self.next())

    def next(self):
        """Returns the next element in the sequence.

        Returns:
            The next element in the sequence.
        """
        self._mutex.acquire()
        result = self._next()
        self._mutex.release()
        return result

    def get_sequence_length(self):
        """Returns the sequence length."""
        return len(self._sequence)

    @abc.abstractmethod
    def reset(self):
        """Resets the iterator for deterministic iteration."""
        pass

    @abc.abstractmethod
    def _next(self):
        """Returns the next element in the sequence.

        Returns:
            The next element in the sequence.
        """


class SequentialIterator(IteratorBase):
    """Iterates over the data in epochs."""

    def __init__(self, sequence):
        """Initializes a new instance of the SequentialIterator class.

        Args:
            sequence: The sequence to iterate over.
        """
        super(SequentialIterator, self).__init__(sequence)
        self._index = 0

    def reset(self):
        """Resets the iterator for deterministic iteration."""
        self._index = 0

    def _next(self):
        """Returns the next element in the sequence.

        Returns:
            The next element in the sequence.
        """
        if self._index == len(self._sequence):
            self.reset()

        result = self._sequence[self._index]
        self._index += 1
        return result


class RandomIterator(IteratorBase):
    """Iterates over the data randomly in epochs."""

    def __init__(self, sequence):
        """Initializes a new instance of the RandomIterator class.

        Args:
            sequence: The sequence to iterate over.
        """
        super(RandomIterator, self).__init__(sequence)
        self._index = 0
        self._order = np.arange(len(sequence))
        self._shuffle()

    def _shuffle(self):
        """Shuffles the current iteration order."""
        np.random.shuffle(self._order)
        self._index = 0

    def reset(self):
        """Resets the iterator for deterministic iteration."""
        self._order = np.arange(len(self._sequence))
        self._index = 0

    def _next(self):
        """Returns the next element in the sequence.

        Returns:
            The next element in the sequence.
        """
        if self._index == len(self._sequence):
            self._shuffle()

        result = self._sequence[self._order[self._index]]
        self._index += 1
        return result


class WeightedRandomIterator(IteratorBase):
    """Randomly samples elements according to a given probability."""

    def __init__(self, sequence, weights):
        """Initializes a new instance of the WeightedRandomIterator class.

        Args:
            sequence: The sequence to iterate over.
            weights: The weight of each element in the sequence.
        """
        super(WeightedRandomIterator, self).__init__(sequence)

        # Make sure that the weights define a probability distribution.
        weights += np.min(weights)
        weights /= np.sum(weights)
        self._weights = weights
        self._indices = np.arange(len(self._sequence))

    def reset(self):
        """Resets the iterator for deterministic iteration."""
        pass

    def _next(self):
        """Returns the next element in the sequence.

        Returns:
            The next element in the sequence.
        """
        index = np.random.choice(self._indices, p=self._weights)
        return self._sequence[index]


@six.add_metaclass(abc.ABCMeta)
class AugmentorBase(object):
    """Augments the data."""

    @abc.abstractmethod
    def augment(self, image, target):
        """Augments the data.

        Args:
            image: The image.
            target: The target image.

        Returns:
            A tuple of augmented image and target image.
        """
        pass


class SubsampleAugmentor(AugmentorBase):
    """Subsamples the image and the target."""

    def __init__(self, factor):
        """Initializes a new instance of the SubsampleAugmentor class.

        Args:
            factor: The sampling factor.
        """
        super(SubsampleAugmentor, self).__init__()
        self._factor = factor

    def augment(self, image, target):
        """Augments the data.

        Args:
            image: The image.
            target: The target image.

        Returns:
            A tuple of augmented image and target image.
        """
        return self._scale_image(image), self._scale_target(target)

    def _scale_image(self, image):
        """Downscales the image."""
        size = image.shape[1] // self._factor, image.shape[0] // self._factor
        return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

    def _scale_target(self, target):
        """Downscales the target.

        This script is based on the following code:
        https://github.com/VisualComputingInstitute/cityscapes-util/blob/master/
        __init__.py

        Copyright (c) Visual Computing Institute RWTH Aachen University
        """
        fy, fx = self._factor, self._factor
        H, W = target.shape
        h, w = H // fy, W // fx

        m = np.min(target)
        M = np.max(target)
        if m == M:
            M = m + 1

        assert -1 <= m, "Labels should not have values below -1"

        # Count the number of occurences of the labels in each "fy x fx cell"
        label_sums = np.zeros((h, w, M + 2))
        mx, my = np.meshgrid(np.arange(w), np.arange(h))
        for dy in range(fy):
            for dx in range(fx):
                label_sums[my, mx, target[dy::fy, dx::fx]] += 1

        # "Don't know" don't count.
        label_sums = label_sums[:, :, :-1]

        # Use the highest-occurence label.
        new_targets = np.argsort(label_sums, 2)[:, :, -1].astype("uint8")

        # But turn "uncertain" cells into "don't know" label.
        counts = label_sums[my, mx, new_targets]
        hit_counts = np.sum(label_sums, 2) * 0.25
        new_targets[counts <= hit_counts] = 255

        return new_targets


class TranslationAugmentor(AugmentorBase):
    """Translates the image randomly."""

    def __init__(self, offset=40):
        """Initializes a new instance of the TranslationAugmentor class.

        Args:
            offset: The offset by which the image is randomly translated.
            p:
        """
        self._offset = offset

    def augment(self, image, target):
        """Augments the data.

        Args:
            image: The image.
            target: The target image.

        Returns:
            A tuple of augmented image and target image.
        """
        # Sample an offset in each direction.
        offsets = np.random.randint(-self._offset, self._offset + 1, (2, ))

        # Extract the image and the label
        image = self._translate_image(image, offsets)
        target = self._translate_target(target, offsets)

        return image, target

    def _translate_image(self, image, offsets):
        """Translates the image and uses reflection padding.

        Args:
            image: The image to translate.
            offsets: The offset in each direction.

        Returns:
            The translated image.
        """
        # Extract the image region that is defined by the offset.
        region = image[
                 max(-offsets[0], 0):image.shape[0] - max(0, offsets[0]),
                 max(-offsets[1], 0):image.shape[1] - max(0, offsets[1]),
                 :]

        # Pad the image using reflection padding.
        padding = (
            (max(0, offsets[0]), max(0, -offsets[0])),
            (max(0, offsets[1]), max(0, -offsets[1])),
            (0, 0)
        )

        region = np.pad(region, padding, "reflect")

        return region

    def _translate_target(self, target, offsets):
        """Translates the image and uses constant -1 padding.

        Args:
            target: The target to translate.
            offsets: The offset in each direction.

        Returns:
            The translated image.
        """
        new_target = -1 * np.ones_like(target)
        new_target[
            max(0, offsets[0]):target.shape[0] + min(0, offsets[0]),
            max(0, offsets[1]):target.shape[1] + min(0, offsets[1])] = target[
                max(-offsets[0], 0):target.shape[0] - max(0, offsets[0]),
                max(-offsets[1], 0):target.shape[1] - max(0, offsets[1])]
        return new_target


class GammaAugmentor(AugmentorBase):
    """Performs random gamma augmentation."""

    def __init__(self, gamma_range=0.1):
        """Initializes a new instance of the GammaAugmentor class.

        Args:
            gamma_range: The range from which to sample gamma.
        """
        self._gamma_range = gamma_range
        assert 0.0 <= self._gamma_range <= 0.5, "Invalid gamma parameter."

    def augment(self, image, target):
        """Augments the data.

        Args:
            image: The image.
            target: The target image.

        Returns:
            A tuple of augmented image and target image.
        """
        # Sample a gamma factor.
        gamma = np.random.uniform(-self._gamma_range, self._gamma_range)

        # Apply the non-linear transformation
        gamma = np.log(
            0.5 + 1 / np.sqrt(2) * gamma) / np.log(0.5 - 1 / np.sqrt(2) * gamma)

        # Perform the gamma correction.
        image **= gamma

        return image, target


class CropAugmentor(AugmentorBase):
    """Randomly extracts crops from the image."""

    def __init__(self, unused_size, unsued_num_classes):
        """Initializes a new instance of the CropAugmentor class."""

    def augment(self, image, target):
        """Augments the data.

        Args:
            image: The image.
            target: The target image.

        Returns:
            A tuple of augmented image and target image.
        """
        raise NotImplementedError("Crop augmentation is not available in "
                                  "Python. Please install the chianti C++ "
                                  "library.")


class RotationAugmentor(AugmentorBase):
    """Randomly rotates the image."""

    def __init__(self, max_angel):
        """Initializes a new instance of the RotationAugmentor class.

        Args:
            max_angel: The maximum angel by which the image is rotated.
        """
        self._max_angel = max_angel

    def augment(self, image, target):
        """Augments the data.

        Args:
            image: The image.
            target: The target image.

        Returns:
            A tuple of augmented image and target image.
        """
        # Sample the rotation factor.
        factor = np.random.uniform(-self._max_angel, self._max_angel)
        if factor < 0:
            factor += 360.0

        # Get the rotation matrix.
        h, w = image.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), factor, 1)

        image = cv2.warpAffine(image, m, (w, h))
        target = cv2.warpAffine(
            target, m, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        return image, target


class ZoomingAugmentor(AugmentorBase):
    """Randomly zooms into or out of the image."""

    def __init__(self, max_factor):
        """Initializes a new instance of the ZoomingAugmentor class.

        Args:
            max_factor: The maximum angel by which the image is rotated.
        """
        self._max_angel = max_factor

    def augment(self, image, target):
        """Augments the data.

        Args:
            image: The image.
            target: The target image.

        Returns:
            A tuple of augmented image and target image.
        """
        raise NotImplementedError("Zooming augmentation is only available in "
                                  "the C++ Chianti library.")


class SaturationAugmentor(AugmentorBase):
    """Randomly alters the image saturation."""

    def __init__(self, min_delta, max_delta):
        """Initializes a new instance of the SaturationAugmentor class.

        Args:
            min_delta: Minimum deviation in the color space.
            max_delta: Maximum deviation in the color space.
        """
        self._min_delta = min_delta
        self._max_delta = max_delta

    def augment(self, image, target):
        """Augments the data.

        Args:
            image: The image.
            target: The target image.

        Returns:
            A tuple of augmented image and target image.
        """
        # Sample the color factor.
        factor = np.random.uniform(self._min_delta, self._max_delta)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_image[:, :, 1] *= factor
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0.0, 1.0)

        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return image, target


class HueAugmentor(AugmentorBase):
    """Randomly alters the image hue."""

    def __init__(self, min_delta, max_delta):
        """Initializes a new instance of the HueAugmentor class.

        Args:
            min_delta: Minimum deviation in the color space.
            max_delta: Maximum deviation in the color space.
        """
        self._min_delta = min_delta
        self._max_delta = max_delta

    def augment(self, image, target):
        """Augments the data.

        Args:
            image: The image.
            target: The target image.

        Returns:
            A tuple of augmented image and target image.
        """
        # Sample the color factor.
        factor = np.random.uniform(self._min_delta, self._max_delta)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_image[:, :, 0] += factor

        # Make sure the values are in [-360, 360].
        hsv_image[:, :, 0] += 360 * (hsv_image[:, :, 0] < 360)
        hsv_image[:, :, 0] -= 360 * (hsv_image[:, :, 0] > 360)

        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return image, target


class CombinedAugmentor(AugmentorBase):
    """Combines multiple augmentors into once."""

    def __init__(self, augmentors):
        """Initializes a new instance of the CombinedAugmentor class.

        Args:
            augmentors: A list of augmentors.
        """
        self._augmentors = augmentors

    def augment(self, image, target):
        """Augments the data.

        Args:
            image: The image.
            target: The target image.

        Returns:
            A tuple of augmented image and target image.
        """
        for augmentor in self._augmentors:
            image, target = augmentor.augment(image, target)

        return image, target

