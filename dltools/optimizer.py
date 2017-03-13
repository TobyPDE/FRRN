import numpy as np
import theano.tensor as T
import time
from . import utility


class MiniBatchOptimizer(object):
    """
    This class optimizes a neural network using Mini batches.
    """

    def __init__(self, train_fn, data_provider, hooks):
        """
        Initializes a new instance of the MiniBatchOptimizer class.

        :param train_fn: The function that updates the network parameters. It has to return a list of losses.
        :param data_provider: A single data provider.
        :param hooks: A list of hooks that are called at certain points in the training process.
        """
        self.train_fn = train_fn
        self.data_provider = data_provider
        self.hooks = hooks

    def call_hooks(self, **kwargs):
        """
        Calls all hooks that have been registered.

        :param kwargs: A list of parameters that is passed to all hooks.
        """
        for h in self.hooks:
            try:
                h.update(**kwargs)
            except Exception as e:
                print("Cannot run hook %s" % h)
                print("Error message: %s" % e)

    def optimize(self):
        """
        Runs the main optimization loop until it is interrupted by the user via CTRL+C.
        """
        update_counter = 0

        # Start the optimization
        with utility.Uninterrupt() as u:
            while not u.interrupted:
                losses = None
                duration = 0

                for i in range(10):
                    # Advance the data_source iterators
                    # Gather the arguments for the training function
                    start = time.time()
                    data = self.data_provider.next()
                    data_duration = time.time() - start

                    update_counter += 1
                    start = time.time()
                    losses = self.train_fn(data.imgs, data.targets, update_counter)
                    duration = time.time() - start
                    print("data: %fs, update: %fs" % (data_duration, duration))

                self.call_hooks(
                    update_counter=update_counter,
                    losses=losses,
                    runtime=duration)

