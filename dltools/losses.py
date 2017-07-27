"""Contains loss definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import theano.tensor as T


def bootstrapped_xentropy(predictions, targets, batch_size=3, multiplier=64):
    """A categorical cross entropy loss for 4D tensors.

    We assume the following layout: (batch, classes, height, width)

    Args:
        predictions: The output of a log softmax layer.
        targets: The predictions as a one-hot encoded tensor.
        batch_size: The batch size
        multiplier: A multiplier variable that determine the number of pixels to
            select in the bootstrapping process. The total number of pixels is
            determined as 512 * multiplier.

    Returns:
        The pixel-bootstrapped cross entropy loss.
    """
    # Compute the pixel-wise cross entropy. Recall that the predictions are
    # already the log softmax.
    xentropy = -T.sum(predictions * targets, axis=1)

    # For each element in the batch, collect the top K worst predictions
    K = 512 * multiplier

    result = T.constant(0, dtype="float32")
    for i in range(batch_size):
        batch_erors = xentropy[i]

        # Void pixels already get a loss of 0, so they're never selected.
        flat_errors = T.flatten(batch_erors)

        # Get the worst predictions.
        worst_errors = T.sort(flat_errors)[-K:]

        result += T.mean(worst_errors)

    result /= T.constant(batch_size, dtype="float32")

    return result

