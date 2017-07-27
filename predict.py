"""Shows the predictions of a FRRN model on the CS validation set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import cv2
from dltools import utility
import train

BATCH_SIZE = 1


def main():
    """Shows the predictions of a FRRN model on the CS validation set."""
    parser = argparse.ArgumentParser(
        description="Shows the predictions of a Full-Resolution Residual"
                    " Network on the Cityscapes validation set.")

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

    parser.add_argument("--cs_folder",
                        type=str,
                        required=True,
                        help="The folder that contains the Cityscapes Dataset.")

    parser.add_argument("--sample_factor",
                        type=int,
                        default=0,
                        help="The sampling factor.")

    args = parser.parse_args()

    # Define the network lasagne graph and try to load the model file
    if args.architecture == "frrn_a":
        sample_factor = 4
    else:
        sample_factor = 2

    if args.sample_factor != 0:
        sample_factor = args.sample_factor

    network = train.define_network(args.architecture, BATCH_SIZE, sample_factor)
    network.load_model(args.model_file)

    val_fn = train.compile_validation_function(network, BATCH_SIZE)
    provider = train.get_validation_provider(args.cs_folder,
                                             sample_factor,
                                             BATCH_SIZE)

    for i in range(provider.get_num_batches()):
        batch = provider.next()
        predictions, loss = val_fn(batch[0], batch[1])

        # Obtain a prediction
        pred_img = utility.create_color_label_image(predictions[0])
        gt_img = utility.create_color_label_image(batch[1][0])
        image = utility.tensor2opencv(batch[0][0])

        logging.info("Image {}. Loss={}".format(i, loss))
        cv2.imshow("Image", image)
        cv2.imshow("Ground Truth", gt_img)
        cv2.imshow("Prediction", pred_img)
        cv2.waitKey()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                        level=logging.DEBUG)
    main()
