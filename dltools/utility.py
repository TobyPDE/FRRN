import os.path
import signal
import numpy as np
import time
import theano.tensor as T
from functools import reduce as _reduce


class Uninterrupt(object):
    """
    Allows you to protect a section of code from being interrupted by the user via CTRL+C.

    Use as follows:

    with Uninterrupt() as u:
        while not u.interrupted:
            # Do something

    via u.interrupted you can check whether the signal has been received once  (i.e. if the user has tried to
    interrupt the process).

    Based on https://gist.github.com/nonZero/2907502
    """
    def __init__(self, sig=signal.SIGINT):
        """Initializes a new instance of the Uninterrupt class.

        :param sig: The signal that the class listens to. SIGINT for CTRL+C.
        """
        self.released = True
        self.interrupted = False
        self.sig = sig

    def __enter__(self):
        """
        Enters the critical section of code. If the signal is received twice, then we
        leave the critical section.
        :return:
        """
        self.released = False
        self.interrupted = False

        # Temporary disable to default signal handler
        self.orig_handler = signal.getsignal(self.sig)

        # Create a new signal handler that waits for the signal to be received twice
        def new_signal_handler(signum, frame):
            # Restore the original signal handler
            self.release()
            # Memorize that the signal has been received once
            self.interrupted = True

        # Register the new signal handler
        signal.signal(self.sig, new_signal_handler)

        return self

    def __exit__(self, type_, value, tb):
        """
        Leaves the critical section and restores the original signal handler.
        """
        self.release()

    def release(self):
        """
        Restores the original signal handler.
        """
        # Have we already restored the original signal handler?
        if not self.released:
            # Nope, do it
            signal.signal(self.sig, self.orig_handler)
            # Don't "restore" it twice.


class Timer(object):
    """
    A simple timer class that let's you measure the execution time of a specific block of code.

    Use as:

    with Timer() as t:
        # do something
    t.interval
    """
    def __init__(self):
        self.interval = 0
        self.running = False

    def __enter__(self):
        """
        Enters the critical section whose runtime is measured.
        """
        self.start = time.clock()
        self.running = True
        return self

    def __exit__(self, *args):
        """
        Leaves the critical section.
        """
        self.end = time.clock()
        self.running = False
        self.interval = self.end - self.start


class VerboseTimer(Timer):
    """
    A timer class that prints the time to stdout.
    """

    def __init__(self, text):
        """
        Initializes a new instance of the VerboseTimer class.

        :param text: The title of the block.
        """
        super().__init__()
        self.text = text

    def __enter__(self):
        """
        Enters the critical section whose runtime is measured.
        """
        super().__enter__()
        print("%s..." % self.text, end="", flush=True)
        return self

    def __exit__(self, *args):
        """
        Leaves the critical section.
        """
        super().__exit__(*args)
        print(" [%.2fs]" % self.interval)


def load_np_data_array(filename):
    """
    Loads a dataset from a numpy file.
    :param filename: The filename of the numpy array.
    :param dtype: The dtype of the data.
    :return: The actual numpy array.
    """
    with np.load(filename) as data:
        return data['arr_0']


def _prod(l):
    """
    Computes the product of the list entries.

    :param l: The list to compute the product over.
    :return: The product of the list entries.
    """
    return _reduce(lambda x, y: x * y, l, 1)


def bootstrapped_categorical_cross_entropy4d_loss(predictions, targets, batch_size=3, multiplier=64):
    """
    A 4d categorical cross entropy loss for 4D tensors. We assume the following layout:
    (batch, classes, height, width)

    This code is based on this implementation of the cross entropy loss:
    https://github.com/lucasb-eyer/DeepFried2/blob/master/DeepFried2/criteria/ClassNLLCriterion.py

    :param predictions: The output of the neural network
    :param targets: The predictions (integer). Layout: (class label, height, width)
    :return: The pixel-bootstrapped cross entropy loss
    """

    # Build the indexing operation that corresponds to the N-dimensional
    # generalization of the classical y[arange(N), T]
    # Also build up the shape of the result at the same time.
    idx = []
    sha = []

    for i in range(4):
        if i == 1:
            idx.append(targets.flatten())
        else:
            nbefore = _prod(predictions.shape[j] for j in range(i) if j != 1)
            nafter = _prod(predictions.shape[j] for j in range(i + 1, 4) if j != 1)
            idx.append(T.tile(T.arange(predictions.shape[i]).repeat(nafter), nbefore))
            sha.append(predictions.shape[i])

    p_y = predictions[tuple(idx)].reshape(sha)

    result = np.float32(0)
    for i in range(batch_size):
        pp = T.flatten(p_y[i])
        tt = T.flatten(targets[i])

        # Select the predictions that are not labeled void
        pp = pp[T.neq(tt, -1).nonzero()]
        pp = T.sort(pp)

        result += -T.mean(pp[:512 * multiplier])

    result /= np.float32(batch_size)

    return result


def print_conf_matrix(conf_matrix, target_names, num_labels):
    """
    Prints a confusion matrix to the console.
    """

    def print_num_value(f):
        colors = ["on_blue", "on_cyan", "on_green", "on_yellow", "on_red"]
        for i in range(len(colors)):
            if f <= (i + 1) * 1 / len(colors):
                print(colored(" %1.2f " % f, "white", colors[i]), end="")
                break

    def print_num_value2(f):
        colors = ["on_blue", "on_cyan", "on_green", "on_yellow", "on_red"]
        for i in range(len(colors)):
            if f <= (i + 1) * 1 / len(colors):
                print(colored(" %04.1f " % (f * 100), "white", colors[i]), end="")
                break

    # Compute the IoU score per class
    class_iou = [conf_matrix[i, i] / (np.sum(conf_matrix[:, i]) + np.sum(conf_matrix[i, :]) - conf_matrix[i, i]) for i in
                 range(0, num_labels)]

    # Compute the pixel accuracy
    accuracy = np.diag(conf_matrix).sum() / conf_matrix.sum()

    # Normalize the matrix
    cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    # cm_normalized = cm_normalized[:-1, :-1]
    # Compute the mean class accuracy
    mean_class_accuracy = np.mean(np.diag(cm_normalized))

    # Print the matrix
    for i in range(num_labels):
        print("%13s | " % (target_names[i],), end="")
        print_num_value2(class_iou[i])
        print(" | ", end="")

        for j in range(num_labels):
            print_num_value2(cm_normalized[i, j])

        print("")
        """
        total_length = 13 + 2 + 4 + 4 + 19* (4 + 0)

        for k in range(total_length):
            print("-", end="")
        print("")
        """

    print("IoU score: %1.5f" % np.average(class_iou))
    print("Accuracy: %1.5f" % accuracy)
    print("Mean class accuracy: %1.5f" % mean_class_accuracy)


def get_training_class_labels():
    """
    Returns a list of all training class labels.
    :return: A list of training class labels.
    """
    import labels as cs_labels

    # Create map id -> color
    res = ["don't care"]
    for label in cs_labels.labels:
        if label.trainId != 255 and label.trainId != -1:
            res.append(label.name)

    return res


def create_color_label_image(np_array):
    """
    Converts a color image from an id images.

    :param np_array: The id image that contains the label ids
    :return: an RGB image.
    """
    # Import the label tool from the official toolbox
    import labels as cs_labels

    # Create map id -> color
    color_map = {label.trainId: label.color for label in cs_labels.labels}
    color_map[-1] = (0, 0, 0)

    result = np.zeros((np_array.shape[0], np_array.shape[1], 3), dtype='uint8')

    # Convert the image
    for i in range(np_array.shape[0]):
        for j in range(np_array.shape[1]):
            result[i, j, :] = color_map[np_array[i, j]]

    result = result[:, :, ::-1]

    return result


def get_cityscapes_path():
    """
    Returns the path to the cityscapes folder.
    :return:
    """
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
    """
    Returns the path to the cityscapes folder.
    :return:
    """
    value = default
    # Does a file already exist?
    if os.path.exists(filename):
        # Read the path
        with open(filename, "r") as f:
            value = f.read()

    # Ask the user for the actual path
    user_input = input("%s [%s]: " % (phrase, value))

    # Did the user enter something?
    if user_input != "":
        # Yes, update the file
        with open(filename, "w") as f:
            f.write(user_input)
        value = user_input

    return value


def tensor2opencv(image):
    """
    Converts an image from the tensor representation for Theano to the OpenCv representation.
    :param image: The image to convert.
    :return: The converted image.
    """
    image = np.rollaxis(image, 0, 3)
    return image[:, :, ::-1]


def opencv2tensor(image):
    """
    Converts an image from the opencv representation to the tensor for Theano representation.
    :param image: The image to convert.
    :return: The converted image.
    """
    return np.rollaxis(image[:,:,::-1], 2)


