import lasagne
import numpy as np
from sklearn.metrics import confusion_matrix


class LoggingHook(object):
    """
    This hook writes information to a log file.
    """

    def __init__(self, logger):
        """
        Initializes a new instance of the LoggingHook class.

        :param logger: A logger.
        """
        self.logger = logger

    def update(self, **kwargs):
        """
        Performs the hook
        :param kwargs: A list of named arguments.
        """

        self.logger.log(
            key="status",
            message="-> Done performing update (%.2fs)" % kwargs["runtime"]
        )
        self.logger.log(
            key="update_runtime",
            message=kwargs["runtime"]
        )
        self.logger.log(
            key="losses",
            message=np.asarray(kwargs["losses"])
        )


class SnapshotHook(object):
    """
    This hook creates regular snapshots of the network.
    """
    def __init__(self, filename, network, frequency=300):
        """
        Initializes a new instance of the SnapshotHook class.

        :param filename: The base filename of the model.
        :param network: The network instance to store.
        :param frequency: The snapshot frequency.
        """
        self.filename = filename
        self.network = network
        self.frequency = frequency

    def update(self, **kwargs):
        """
        Performs the hook.

        :param kwargs:
        :return:
        """
        # Run the hook now?
        if kwargs["update_counter"] % self.frequency == 0:
            # Yes
            np.savez(
                "%s_snapshot_%d.npz" % (self.filename, kwargs["update_counter"]),
                *lasagne.layers.get_all_param_values(self.network))


class SegmentationValidationHook(object):
    """
    Performs a validation run for semantic segmentation at certain times in the training process. Uses a data provider
    in order to
    """
    def __init__(self, val_fn, data_provider, logger, frequency=300, num_classes=19):
        """
        Initializes a new instance of the ValidationHook class.

        :param val_fn: A function that returns the predictions for each image and a list of losses.
        :param data: A tuple of (images, targets).
        :param batch_size: The batch size.
        :param logger: A logger instance.
        :param frequency: The validation frequency.
        """
        self.val_fn = val_fn
        self.data_provider = data_provider
        self.logger = logger
        self.frequency = frequency
        self.num_classes = num_classes

    def update(self, **kwargs):
        """
        Runs the validation function hook.
        """

        if kwargs["update_counter"] % self.frequency == 0 and kwargs["update_counter"] > 0:
            self.logger.log(
                key="validation_checkpoint",
                message=kwargs["update_counter"]
            )
            self.logger.log(
                key="status",
                message="-> Start validation run"
            )

            # Run the validation
            # Create the confusion matrix
            conf_matrix = np.zeros((self.num_classes, self.num_classes)).astype('int64')

            accumulated_loss = 0

            for batch_counter in range(self.data_provider.get_num_batches()):
                self.logger.log(
                    key="status",
                    message="--> Validate batch %d/%d" % (batch_counter + 1, self.data_provider.get_num_batches())
                )
                print("--> Validate batch %d/%d" % (batch_counter + 1, self.data_provider.get_num_batches()))

                self.data_provider.next()
                images, targets = self.data_provider.current()

                predictions, loss = self.val_fn(images, targets)

                accumulated_loss += loss

                # Mark the don't care predictions
                # Flatten the predictions and targets
                flat_predictions = predictions.flatten()
                flat_targets = targets.flatten()
                # Get the position of the don't cares
                mask = flat_targets != -1
                # Select the non-don't cares
                flat_targets = flat_targets[mask]
                flat_predictions = flat_predictions[mask]

                conf_matrix += confusion_matrix(flat_targets, flat_predictions, labels=list(range(self.num_classes))).astype('int64')

            accumulated_loss /= self.data_provider.get_num_batches()

            self.logger.log(
                key="conf_matrix",
                message=conf_matrix
            )
            self.logger.log(
                key="validation_loss",
                message=accumulated_loss
            )
