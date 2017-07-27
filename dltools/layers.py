"""Definition of custom lasagne layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import theano
import theano.tensor as T
from lasagne import layers


class BilinearUpscaleLayer(layers.Layer):
    """Upscales tensors via bilinear interpolation.

    This layer upscales the 4D input tensor along the trailing spatial
    dimensions using bilinear interpolation. You have to specify image
    dimensions in order to use this layer - even if you want to have a
    fully convolutional network.
    """

    def __init__(self, incoming, factor, **kwargs):
        """Initializes a new instance of the BilinearUpscaleLayer class.

        Args:
            incoming: The incoming network stream
            factor: The factor by which to upscale the input
        """
        super(BilinearUpscaleLayer, self).__init__(incoming, **kwargs)
        self.factor = factor

    def get_output_shape_for(self, input_shape):
        """Computes the output shape of the layer given the input shape.

        Args:
            input_shape: The input shape

        Returns:
            The output shape.
        """
        return (input_shape[0],
                input_shape[1],
                self.factor * input_shape[2],
                self.factor * input_shape[3])

    def get_output_for(self, input, **kwargs):
        """Constructs the Theano graph for this layer.

        Args:
            input: Symbolic input variable

        Returns:
            Symbolic output variable
        """
        return T.nnet.abstract_conv.bilinear_upsampling(input, self.factor)


class BatchNormLayer(layers.BatchNormLayer):
    """Extension of the BN layer that collects the updates in a dict."""

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        # If the BN vars shall be updates as before, redirect to the parent
        # implementation.
        if not isinstance(batch_norm_update_averages, dict):
            return super(BatchNormLayer, self).get_output_for(
                input, deterministic, batch_norm_use_averages,
                batch_norm_update_averages, **kwargs)
        else:
            input_mean = input.mean(self.axes)
            input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

            # Decide whether to use the stored averages or mini-batch statistics
            if batch_norm_use_averages is None:
                batch_norm_use_averages = deterministic
            use_averages = batch_norm_use_averages

            if use_averages:
                mean = self.mean
                inv_std = self.inv_std
            else:
                mean = input_mean
                inv_std = input_inv_std

            # Instead of automatically updating the averages, we add the update
            # ops to a dictionary.
            update_averages = batch_norm_update_averages
            if isinstance(update_averages, dict):
                update_averages[self.mean] = ((1 - self.alpha) * self.mean +
                                              self.alpha * input_mean)
                update_averages[self.inv_std] = ((1 - self.alpha) *
                                                 self.inv_std + self.alpha *
                                                 input_inv_std)

            # prepare dimshuffle pattern inserting broadcastable axes as needed
            param_axes = iter(range(input.ndim - len(self.axes)))
            pattern = ['x' if input_axis in self.axes
                       else next(param_axes)
                       for input_axis in range(input.ndim)]

            # apply dimshuffle pattern to all parameters
            beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
            gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
            mean = mean.dimshuffle(pattern)
            inv_std = inv_std.dimshuffle(pattern)

            # normalize
            normalized = (input - mean) * (gamma * inv_std) + beta
            return normalized
