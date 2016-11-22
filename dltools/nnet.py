import os
import numpy as np
import lasagne
import theano.tensor as T


class NeuralNetwork(object):
    """
    This is the abstract base class of the neural network.
    """

    def __init__(self):
        self.input_layers = []
        """The input layers of the neural network."""
        self.output_layers = []
        """The output layers of the neural network."""
        self.batchnorm_layers = []
        """These are the batch normalization layers of the networks in topological order."""
        self.splits = []
        """These are the split points of the neural network. The gradients are computed in-between the split points"""

    @staticmethod
    def join_outputs(outputs):
        """
        Joins several network outputs in order to have a common output for setting parameter values.

        :param outputs: A list of lasagne layers.
        :return: A lasagne layer that joins the outputs.
        """
        # Make all output of the same size
        outputs = [lasagne.layers.ExpressionLayer(layer, lambda Y: T.sum(Y), output_shape='auto') for layer in outputs]
        return lasagne.layers.ElemwiseSumLayer(outputs)

    def load_model(self, filename):
        """
        Loads the parameters from a file.
        :param filename: The model file.
        """
        # Join all outputs
        joined_network = NeuralNetwork.join_outputs(self.output_layers[0:1])

        if os.path.isfile(filename):
            with np.load(filename) as data:
                lasagne.layers.set_all_param_values(joined_network, [data['arr_' + str(a)] for a in range(len(data.files))])

    def save_model(self, filename):
        """
        Stores the network parameters in a model file.
        :param filename: The model filename.
        """
        # Join all outputs
        joined_network = NeuralNetwork.join_outputs(self.output_layers)
        np.savez(filename, *lasagne.layers.get_all_param_values(joined_network))

    def get_params(self, **kwargs):
        """
        Returns a list of all network parameters.
        :return: A list of all network parameters.
        """
        return lasagne.layers.get_all_params(self.output_layers, **kwargs)

    def get_num_params(self):
        """
        Count the total number of parameters of the network.
        :return: The total number of parameters.
        """
        params = lasagne.layers.get_all_param_values(self.output_layers, trainable=True)
        return sum([len(p.flatten()) for p in params])

