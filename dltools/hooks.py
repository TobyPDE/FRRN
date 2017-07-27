"""Defines hooks that can run during training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lasagne
import numpy as np
from sklearn import metrics


class LoggingHook(object):
    """This hook writes information to a log file."""

    def __init__(self, logger):
        """Initializes a new instance of the LoggingHook class.

        Args:
            logger: A logger instance.
        """
        self._logger = logger

    def update(self, **kwargs):
        """Executes the hook.

        Args:
            **kwargs: Optimizer state dictionary.
        """

        self._logger.log(
            key="status",
            message="Log at iteration %d" % kwargs["update_counter"]
        )

        self._logger.log(
            key="update_counter",
            message=kwargs["update_counter"]
        )

        self._logger.log(
            key="update_runtime",
            message=kwargs["runtime"]
        )
        self._logger.log(
            key="losses",
            message=np.asarray(kwargs["losses"])
        )


class SnapshotHook(object):
    """Hook for storing snapshots of the network's weights."""

    def __init__(self, filename, network, interval):
        """Initializes a new instance of the SnapshotHook class.

        Args:
            filename: The base filename of the model.
            network: The network instance to store.
            interval: The snapshot interval.
        """
        self._filename = filename
        self._network = network
        self._interval = interval

    def update(self, **kwargs):
        """Executed the hook.

        Args:
            **kwargs: The optimizer dictionary.
        """
        # Run the hook now?
        if kwargs["update_counter"] % self._interval == 0:
            # Yes
            np.savez(
                "%s_snapshot_%d.npz" % (
                    self._filename, kwargs["update_counter"]),
                *lasagne.layers.get_all_param_values(self._network))


class SegmentationValidationHook(object):
    """Performs a validation run for semantic segmentation."""

    def __init__(self, val_fn, data_provider, logger, interval=300,
                 num_classes=19):
        """Initializes a new instance of the SegmentationValidationHook class.

        Args:
            val_fn: A function that returns the predictions for each image and
            a list of losses.
            data_provider: A chianti data provider.
            logger: A logger instance.
            interval: The validation interval.
        """
        self._val_fn = val_fn
        self._data_provider = data_provider
        self._logger = logger
        self._interval = interval
        self._num_classes = num_classes

    def update(self, **kwargs):
        """Runs the validation hook."""

        update_now = kwargs["update_counter"] % self._interval == 0
        if update_now and kwargs["update_counter"] > 0:
            self._logger.log(
                key="validation_checkpoint",
                message=kwargs["update_counter"]
            )
            self._logger.log(
                key="status",
                message="-> Start validation run"
            )

            # Initialize the confusion matrix
            conf_matrix = np.zeros(
                (self._num_classes, self._num_classes)).astype('int64')

            accumulated_loss = 0

            self._data_provider.reset()
            for batch_counter in range(self._data_provider.get_num_batches()):
                self._logger.log(
                    key="status",
                    message="--> Validate batch %d/%d" % (
                        batch_counter + 1,
                        self._data_provider.get_num_batches()))

                batch = self._data_provider.next()
                images = batch[0]
                targets = batch[1]
                predictions, loss = self._val_fn(images, targets)

                accumulated_loss += loss

                # Mark the don't care predictions
                # Flatten the predictions and targets
                flat_predictions = predictions.flatten()
                non_void_pixels = (np.max(targets, axis=1) != 0.0).flatten()
                flat_targets = np.argmax(targets, axis=1).flatten()

                # Select the non-don't cares
                flat_targets = flat_targets[non_void_pixels]
                flat_predictions = flat_predictions[non_void_pixels]

                conf_matrix += metrics.confusion_matrix(
                    flat_targets,
                    flat_predictions,
                    labels=np.arange(self._num_classes, dtype='int64'))

            accumulated_loss /= self._data_provider.get_num_batches()

            self._logger.log(
                key="conf_matrix",
                message=conf_matrix
            )
            self._logger.log(
                key="validation_loss",
                message=accumulated_loss
            )
