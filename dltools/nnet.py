"""Wrapper class for neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import lasagne


class NeuralNetwork(object):
    """Wrapper class for neural networks."""

    def __init__(self):
        """Initializes a new instance of the NeuralNetwork class."""
        self.input_layers = []
        self.output_layers = []
        self.splits = []

    def load_model(self, filename):
        """
        Loads the parameters from a file.
        :param filename: The model file.
        """
        if os.path.isfile(filename):
            with np.load(filename) as data:
                lasagne.layers.set_all_param_values(
                    self.output_layers,
                    [data['arr_' + str(a)] for a in range(len(data.files))])

    def save_model(self, filename):
        """Stores the network parameters in a model file.

        Args:
            filename: The model filename.
        """
        np.savez(filename, *lasagne.layers.get_all_param_values(
            self.output_layers))

    def get_params(self, **kwargs):
        """Returns a list of all network parameters.

        Args:
            **kwargs: Arguments for `lasagne.layers.get_all_params`.

        Returns:
            List of network parameters.
        """
        return lasagne.layers.get_all_params(self.output_layers, **kwargs)

    def get_num_params(self):
        """Count the total number of parameters of the network.

        Returns:
            The total number of parameters.
        """
        params = lasagne.layers.get_all_param_values(self.output_layers,
                                                     trainable=True)
        return sum([len(p.flatten()) for p in params])

