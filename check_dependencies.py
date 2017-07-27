"""Runs a sequence of tests to determine if all dependencies are installed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import logging
import importlib
import sys


CS_TRAIN_IMGS = 2975
CS_VAL_IMGS = 500


def check_theano():
    """Checks if theano is installed correctly."""
    try:
        import theano
        import theano.tensor as T

        # Check float type.
        if theano.config.floatX != "float32":
            logging.error("Theano float type must be float32. Add "
                          "floatX=float32 to your .theanorc.")
        else:
            logging.info("Theano float is float32.")

        # Check if cudnn softmax is available.
        try:
            from dltools import architectures
            architectures.FRRNBuilderBase.log_softmax_4d(T.ftensor4())
            logging.info("cuDNN spatial softmax found.")
        except:
            logging.error("Cannot create cuDNN spatial log softmax. Install "
                          "cuDNN and make sure that theano uses the GPU.")
    except:
        logging.error("Cannot import theano.")


def check_package(package_name):
    """Checks if numpy is installed.

    Args:
        package_name: The name of the package to import.
    """
    try:
        importlib.import_module(package_name)
        logging.info("Successfully imported {}.".format(package_name))
    except:
        logging.error("Cannot import {}. Please install.".format(package_name))


def check_chianti():
    """Checks if the Chianti C++ library is available."""
    try:
        import pychianti
        logging.info("Use Chianti C++ library.")
    except:
        logging.info("Chianti C++ library is not available. Will default to "
                     "slower Python implementation.")


def _load_cs_split(cs_folder, split):
    """Loads a split from the CityScapes dataset.

    Args:
        cs_folder: The CityScapes folder.
        split: The split to load.
    """
    try:
        from dltools import utility
        dataset = utility.get_image_label_pairs(cs_folder, split)
        return dataset
    except:
        return []


def check_cityscapes(cs_folder):
    """Checks if the Cityscapes dataset is available."""
    train_images = _load_cs_split(cs_folder, "train")
    val_images = _load_cs_split(cs_folder, "val")

    if len(train_images) != CS_TRAIN_IMGS:
        logging.error("Invalid number of CityScapes training "
                      "images {}.".format(len(train_images)))
    else:
        logging.info("Found CityScapes training set.")

    if len(val_images) != CS_VAL_IMGS:
        logging.error("Invalid number of CityScapes validation "
                      "images {}.".format(len(val_images)))
    else:
        logging.info("Found CityScapes validation set.")


def check_python():
    """Issues a warning if you're not running P2.7 or P3.4."""
    version = sys.version[:3]
    if version != "2.7" and version != "3.4":
        logging.warning("You are running Python {}. We only officially support "
                        "Python 2.7 and 3.4. This software may "
                        "or may not run.".format(version))
    else:
        logging.info("Found supported Python version {}.".format(version))


def main(cs_folder):
    """Runs all checks."""
    check_python()
    check_package("numpy")
    check_package("cv2")
    check_package("sklearn")
    check_package("sklearn.metrics")
    check_package("scipy")
    check_package("theano")
    check_package("lasagne")
    check_theano()
    check_chianti()
    check_cityscapes(cs_folder)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description="Checks if all needed dependencies are installed "
                    "correctly.")

    parser.add_argument("--cs_folder",
                        type=str,
                        required=True,
                        help="The folder that contains the Cityscapes Dataset.")

    args = parser.parse_args()

    main(args.cs_folder)
