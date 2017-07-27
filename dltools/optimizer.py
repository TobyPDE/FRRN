"""Defines the main training loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time


class MiniBatchOptimizer(object):
    """Defines the main training loop."""

    def __init__(self, train_fn, data_provider, hooks):
        """Initializes a new instance of the MiniBatchOptimizer class.

        Args:
            train_fn: The function that updates the network parameters.
                It has to return a list of losses.
            data_provider: A single data provider.
            hooks: A list of hooks that are called at certain points in the
                training process.
        """
        self.train_fn = train_fn
        self.data_provider = data_provider
        self.hooks = hooks

    def call_hooks(self, **kwargs):
        """Calls all hooks that have been registered.

        Args:
            **kwargs: A list of parameters that is passed to all hooks.
        """
        for h in self.hooks:
            try:
                h.update(**kwargs)
            except Exception as e:
                logging.info("Cannot run hook {}".format(h))
                logging.info("Error message: {}".format(e))

    def optimize(self):
        """Runs the main loop until it is interrupted  via CTRL+C."""
        update_counter = 0

        # Start the optimization
        while True:
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
                losses = self.train_fn(data[0], data[1], update_counter)
                duration = time.time() - start
                logging.info("data: {}s, update: {}s".format(
                    data_duration, duration))

            self.call_hooks(
                update_counter=update_counter,
                losses=losses,
                runtime=duration)
