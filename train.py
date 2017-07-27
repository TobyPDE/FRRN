"""Trains the FRRN A architecture on the CityScapes dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import functools
import pickle
import lasagne
import logging as pylogging
from dltools import hybrid_training, utility, architectures, logging, optimizer, hooks, losses
import theano
import theano.tensor as T
import numpy as np

# Try to import pychianti and if it doesn't work, import the python adapter.
# This allows us to use the code without installing the C++ library.
try:
    import pychianti
except:
    from dltools import pychianti_adapter as pychianti


NUM_CLASSES = 19
IMAGE_CHANNELS = 3
IMAGE_ROWS = 1024
IMAGE_COLS = 2048
BASE_CHANNELS = 48
FR_CHANNELS = 32
MULTIPLIER = 2
BOOTSTRAP_MULTIPLIER = 64
REDUCE_LR_INTERVAL = 5000


def define_network(arch, batch_size, sample_factor, crop_size=None):
    """Creates the network architecture.

    Args:
        arch: The architecture type. "frrn_a" or "frrn_b"
        batch_size: The batch size.
        sample_factor: The subsampling factor.
        crop_size: The size of the image crops. None will result in full-frame
            training.

    Returns:
        The newly created network instance.

    Raises:
        ValueError: If `arch` is not a valid architecture identifier.
    """
    if arch == "frrn_a":
        builder = architectures.FRRNABuilder
    elif arch == "frrn_b":
        builder = architectures.FRRNBBuilder
    else:
        raise ValueError("Invalid network architecture {}.".format(arch))

    # Define the theano variables
    input_var = T.ftensor4()

    builder = builder(
        base_channels=BASE_CHANNELS,
        lanes=FR_CHANNELS,
        multiplier=MULTIPLIER,
        num_classes=NUM_CLASSES
    )

    if crop_size is None:
        network = builder.build(
            input_var=input_var,
            input_shape=(batch_size,
                         IMAGE_CHANNELS,
                         IMAGE_ROWS // sample_factor,
                         IMAGE_COLS // sample_factor))
    else:
        network = builder.build(
            input_var=input_var,
            input_shape=(batch_size,
                         IMAGE_CHANNELS,
                         crop_size,
                         crop_size))

    return network


def compile_train_function(network, batch_size, learning_rate):
    """Compiles the training function.

    Args:
        network: The network instance.
        batch_size: The training batch size.
        learning_rate: The learning rate.
    Returns:
    The update function that takes a batch of images and targets and updates the
    network weights.
    """
    learning_rate = np.float32(learning_rate)

    input_var = network.input_layers[0].input_var
    target_var = T.ftensor4()

    # Loss function
    loss_fn = functools.partial(
        losses.bootstrapped_xentropy,
        targets=target_var,
        batch_size=batch_size,
        multiplier=BOOTSTRAP_MULTIPLIER
    )

    # Update function
    lr = theano.shared(learning_rate)

    update_fn = functools.partial(lasagne.updates.adam, learning_rate=lr)

    pylogging.info("Compile SGD updates")
    gd_step = hybrid_training.compile_gd_step(
        network, loss_fn, [input_var, target_var], update_fn)

    reduce_lr = theano.function(
        inputs=[],
        updates=collections.OrderedDict([
            (lr, T.maximum(np.float32(5e-5), lr / np.float32(1.25)))
        ])
    )

    def _compute_update(imgs, targets, update_counter):
        if (update_counter + 1) % REDUCE_LR_INTERVAL == 0:
            reduce_lr()
        loss = gd_step(imgs, targets)

        return loss

    return _compute_update


def compile_validation_function(network, batch_size):
    """Compiles the validation function.

    Args:
        network: The network instance.
        batch_size: The batch size.

    Returns:
    A function that takes in a batch of images and targets and returns the
    predicted segmentation mask and the loss.
    """
    input_var = network.input_layers[0].input_var
    target_var = T.ftensor4()

    predictions = lasagne.layers.get_output(
        network.output_layers, deterministic=True)[0]

    loss = losses.bootstrapped_xentropy(
        predictions=predictions,
        targets=target_var,
        batch_size=batch_size,
        multiplier=BOOTSTRAP_MULTIPLIER
    )

    pylogging.info("Compile validation function")
    return theano.function(
        inputs=[input_var, target_var],
        outputs=[T.argmax(predictions, axis=1), loss]
    )


def get_training_provider(cityscapes_folder,
                          sample_factor,
                          batch_size,
                          iterator_type,
                          crop_size):
    """Creates the training data provider.

    Args:
        cityscapes_folder: The folder in which the Cityscapes Dataset is
            located.
        sample_factor: The image sampling factor.
        batch_size: The batch size.
        iterator_type: The iterator type. "uniform" or "weighted".
        crop_size: The size of the image crops. None will result in full-frame
            training.

    Returns:
    A chianti data provider.
    """
    augmentors = [
        pychianti.Augmentor.Translation(120),
    ]

    if sample_factor > 1:
        augmentors.append(pychianti.Augmentor.Subsample(sample_factor))

    if crop_size is not None:
        augmentors.append(pychianti.Augmentor.Crop(crop_size, NUM_CLASSES))

    augmentors.extend([
        pychianti.Augmentor.Gamma(0.05),
        pychianti.Augmentor.Rotation(10),
        pychianti.Augmentor.Saturation(0.5, 1.5),
        pychianti.Augmentor.Hue(-30, 30),
    ])

    images = utility.get_image_label_pairs(cityscapes_folder, "train")

    if iterator_type == "uniform":
        iterator = pychianti.Iterator.Random(images)
    elif iterator_type == "weighted":
        # Load the image weights
        with open("data_weights.pkl", "rb") as f:
            w = pickle.load(f)

        weights = []
        for img in images:
            image_name = img[0].split("/")[-1]
            weights.append(w[image_name])
        iterator = pychianti.Iterator.WeightedRandom(images, weights)
    else:
        raise ValueError("Invalid iterator type {}.".format(iterator_type))

    provider = pychianti.DataProvider(
        pychianti.Augmentor.Combined(augmentors),
        pychianti.Loader.RGB(),
        pychianti.Loader.ValueMapper(utility.cityscapes_value_map),
        iterator,
        batch_size,
        NUM_CLASSES)

    return provider


def get_validation_provider(cityscapes_folder,
                            sample_factor,
                            batch_size,
                            crop_size=None):
    """Creates the validation data provider.

    Args:
        cityscapes_folder: The folder in which the Cityscapes Dataset is
            located.
        sample_factor: The image sampling factor.
        batch_size: The batch size.
        crop_size: The size of the image crops. None will result in full-frame
            training.

    Returns:
    A chianti data provider.
    """
    augmentors = []

    if sample_factor > 1:
        augmentors.append(pychianti.Augmentor.Subsample(sample_factor))

    if crop_size is not None:
        augmentors.append(pychianti.Augmentor.Crop(crop_size, NUM_CLASSES))

    validation_images = utility.get_image_label_pairs(cityscapes_folder, "val")

    return pychianti.DataProvider(
        pychianti.Augmentor.Combined(augmentors),
        pychianti.Loader.RGB(),
        pychianti.Loader.ValueMapper(utility.cityscapes_value_map),
        pychianti.Iterator.Sequential(validation_images),
        batch_size,
        NUM_CLASSES)


def main():
    """Trains a FRRN architecture on the Cityscapes Dataset."""
    parser = argparse.ArgumentParser(
        description="Trains a Full-Resolution Residual"
                    " Network on the Cityscapes"
                    " Dataset.")

    parser.add_argument("--architecture",
                        type=str,
                        choices=["frrn_a", "frrn_b"],
                        required=True,
                        help="The network architecture type.")

    parser.add_argument("--model_file",
                        type=str,
                        required=True,
                        help="The model filename. Weights are initialized to "
                             "the given values if the file exists. Snapshots "
                             "are stored using a _snapshot_[iteration] "
                             "post-fix.")

    parser.add_argument("--log_file",
                        type=str,
                        required=True,
                        help="The log filename. Use log_monitor.py in order to "
                             "monitor training progress in the terminal.")

    parser.add_argument("--cs_folder",
                        type=str,
                        required=True,
                        help="The folder that contains the Cityscapes Dataset.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=3,
                        help="The batch size.")

    parser.add_argument("--validation_interval",
                        type=int,
                        default=500,
                        help="The validation interval.")

    parser.add_argument("--iterator",
                        type=str,
                        default="uniform",
                        choices=["uniform", "weighted"],
                        help="The dataset iterator type.")

    parser.add_argument("--crop_size",
                        type=int,
                        default=0,
                        help="The size of crops to extract from the "
                             "full-resolution images. If 0, then no crops "
                             "will be extracted.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-3,
                        help="The learning rate to use.")

    parser.add_argument("--sample_factor",
                        type=int,
                        default=0,
                        help="The sampling factor.")

    args = parser.parse_args()

    # Determine the sampling factor based on the network architecture
    if args.architecture == "frrn_a":
        sample_factor = 4
    else:
        sample_factor = 2

    if args.sample_factor != 0:
        sample_factor = args.sample_factor

    if args.crop_size > 0:
        crop_size = args.crop_size
        sample_factor = 1
    else:
        crop_size = None

    pylogging.info("Sample factor: {}".format(sample_factor))

    # Define the network lasagne graph and try to load the model file
    network = define_network(args.architecture,
                             args.batch_size,
                             sample_factor,
                             crop_size)
    pylogging.info("Try to load weights from {}".format(args.model_file))
    network.load_model(args.model_file)

    # Get the logger
    logger = logging.FileLogWriter(args.log_file)

    # Create the optimizer
    opt = optimizer.MiniBatchOptimizer(
        compile_train_function(network, args.batch_size, args.learning_rate),
        get_training_provider(args.cs_folder,
                              sample_factor,
                              args.batch_size,
                              args.iterator,
                              crop_size),
        [
            hooks.SnapshotHook(
                args.model_file,
                network,
                interval=args.validation_interval),
            hooks.LoggingHook(logger),
            hooks.SegmentationValidationHook(
                compile_validation_function(network, args.batch_size),
                get_validation_provider(args.cs_folder,
                                        sample_factor,
                                        args.batch_size,
                                        crop_size),
                logger,
                interval=args.validation_interval)
        ])

    pylogging.info("Start training")
    opt.optimize()


if __name__ == "__main__":
    pylogging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                          level=pylogging.DEBUG)
    main()
