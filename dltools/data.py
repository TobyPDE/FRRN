import numpy as np
from threading import Thread, Lock
import cityscapesutil


class AbstractDataProvider(object):
    """
    This is the base class for all data providers.
    """
    def get_num_batches(self):
        """
        Returns the number of batches that this data provider uses.
        :return:
        """
        return -1


class AbstractThreadedDataProvider(AbstractDataProvider):
    """
    This is the base class for threaded data providers that compute the next patch in a separate thread.
    """
    def __init__(self):
        """
        Initializes a new instance of the AbstractThreadedDataProvider.
        """
        self.next_batch = None
        self.next_batch_mutex = Lock()

    def next(self):
        """
        Increases the current batch counter.
        """
        pass

    def _update_next_batch(self):
        """
        Computes the next batch.
        """
        pass

    def compute_next_batch(self):
        """
        Returns the current mini-batch.
        :return:
        """
        self.next_batch_mutex.acquire()
        self._update_next_batch()
        self.next_batch_mutex.release()

    def current(self):
        """
        Returns the current mini-batch.
        :return:
        """
        self.next_batch_mutex.acquire()
        if self.next_batch is None:
            self._update_next_batch()
            self.next()

        result = self.next_batch
        self.next_batch = None
        self.next_batch_mutex.release()

        # Compute the next batch async
        thread = Thread(target=self.compute_next_batch)
        thread.start()

        return result


class CityscapesHDDDataProvider(AbstractThreadedDataProvider):
    """
    This data provider loads the batches from the hard drive and, therefore, does not rely on pre-processed
    data sets.
    """

    def __init__(self, base_folder, file_folder="train", batch_size=1, sampling_factor=2, threshold=0.5, augmentor=None, random=True):
        """
        Initializes a new instance of the CityscapesHDDDataProvider class.

        :param base_folder: The base folder to load the images from.
        :param batch_size: The batch size.
        :param sampling_factor: The down-sampling factor.
        :param threshold: The label down-sampling threshold.
        """
        super(CityscapesHDDDataProvider, self).__init__()

        # Load the image names
        fine_annotations = cityscapesutil.image_names(base_folder, file_folder)

        # Merge the two
        self.image_names = [(x, file_folder != "train_extra") for x in fine_annotations]
        self.sampling_factor = sampling_factor
        self.threshold = threshold
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.random = random

        self.index = -1
        self.order = np.arange(len(self.image_names))
        if self.batch_size > 0:
            self.num_batches = len(self.image_names) // self.batch_size
        else:
            self.num_batches = 0
        self.shuffle()

    def get_num_batches(self):
        """
        Returns the number of batches that this data provider uses.
        :return:
        """
        return self.num_batches

    def shuffle(self):
        """
        Shuffles the data.
        """
        self.order = np.arange(len(self.image_names))
        if self.random:
            np.random.shuffle(self.order)

    def next(self):
        """
        Increases the current batch counter.
        """
        self.index += 1

        # Are we out of bounds?
        if self.index + 1 >= self.num_batches:
            # Yep, reset
            self.index = 0
            self.shuffle()

    def _update_next_batch(self):
        indices = self.order[self.index * self.batch_size:(self.index + 1) * self.batch_size]

        # Load the images
        images = []
        labels = []

        for i in range(self.batch_size):
            file_name = self.image_names[indices[i]]
            images.extend(cityscapesutil.load_images([file_name[0]], downscale_factor=self.sampling_factor))
            labels.extend(cityscapesutil.load_labels(
                [file_name[0]],
                label_downscale_threshold=self.threshold,
                downscale_factor=self.sampling_factor,
                fine=file_name[1]))

        images = np.asarray(images, dtype='float32')
        labels = np.asarray(labels, dtype='int32')

        result = (images, labels)

        if self.augmentor is not None:
            result = self.augmentor.augment([d for d in result])

        self.next_batch = tuple(result)


class DataProvider(AbstractDataProvider):
    """
    This class let's you randomly iterate over a dataset and apply augmentations as you go.

    TODO: Use Threaded provider as base class.
    """
    def __init__(self, data_arrays, batch_size=1, augmentor=None, random=True):
        """
        Initializes a new instance of the DataTargetsProvider.

        :param data_arrays: The data array.
        :param batch_size: The batch size.
        :param augmentor: The augmentation chain.
        :param random: If the batches shall be shuffled.
        """
        self.data_arrays = data_arrays
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.random = random
        self.next_batch = None
        self.next_batch_mutex = Lock()
        self.order = None

        self.index = -1
        if self.batch_size > 0:
            self.num_batches = len(data_arrays[0]) // self.batch_size
        else:
            self.num_batches = 0
        self.shuffle()

        # If there are more than one data arrays, then they have to have the same length
        if len(self.data_arrays) > 1:
            for i in range(1, len(self.data_arrays)):
                assert len(self.data_arrays[0]) == len(self.data_arrays[i])

    def get_num_batches(self):
        """
        Returns the number of batches that this data provider uses.
        :return:
        """
        return self.num_batches

    def shuffle(self):
        """
        Shuffles the data.
        """
        indices = np.arange(len(self.data_arrays[0]))
        if self.random:
            np.random.shuffle(indices)
        self.order = indices

    def next(self):
        """
        Increases the current batch counter.
        """
        self.index += 1

        # Are we out of bounds?
        if (self.index + 1) * self.batch_size >= len(self.data_arrays[0]):
            # Yep, reset
            self.index = 0
            self.shuffle()

    def _update_next_batch(self):
        indices = self.order[self.index * self.batch_size:(self.index + 1) * self.batch_size]
        result = []
        for i in range(len(self.data_arrays)):
            result.append(self.data_arrays[i][indices])

        if self.augmentor is not None:
            result = self.augmentor.augment([d.copy() for d in result])

        self.next_batch = tuple(result)

    def compute_next_batch(self):
        """
        Returns the current mini-batch.
        :return:
        """
        self.next_batch_mutex.acquire()
        self._update_next_batch()
        self.next_batch_mutex.release()

    def current(self):
        """
        Returns the current mini-batch.
        :return:
        """
        self.next_batch_mutex.acquire()
        if self.next_batch is None:
            self._update_next_batch()

        result = self.next_batch
        self.next_batch = None
        self.next_batch_mutex.release()

        # Compute the next batch async
        thread = Thread(target=DataProvider.compute_next_batch, args=(self,))
        thread.start()

        return result


class PatchDataProvider(AbstractThreadedDataProvider):
    """
    This class let's you randomly iterate over a sampled patches from a pair of input output images.
    """
    def __init__(self, provider, patch_size=(512, 256)):
        """
        Initializes a new instance of the DataTargetsProvider.
        :param provider: The underlying data provider.
        :param patch_size: The size of the patches to sample.
        """
        super(PatchDataProvider, self).__init__()
        self.provider = provider
        self.patch_size = patch_size

    def next(self):
        """
        Increases the current batch counter.
        """
        self.provider.next()

    def _update_next_batch(self):
        # Get the next batch
        x, t = self.provider.current()

        new_x = np.empty((x.shape[0], x.shape[1], *self.patch_size), dtype='float32')
        new_t = np.empty((t.shape[0], *self.patch_size), dtype='int32')

        # Sample patches
        for i in range(x.shape[0]):
            offset_x = np.random.randint(0, x.shape[2] - self.patch_size[0] + 1)
            offset_y = np.random.randint(0, x.shape[3] - self.patch_size[1] + 1)

            new_x[i] = x[i, :, offset_x:offset_x + self.patch_size[0], offset_y:offset_y + self.patch_size[1]]
            new_t[i] = t[i, offset_x:offset_x + self.patch_size[0], offset_y:offset_y + self.patch_size[1]]

        self.next_batch = (new_x, new_t)
