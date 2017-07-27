"""Defines classes for building FRRNs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lasagne
import theano

from dltools import layers
from dltools import nnet


class BuilderBase(object):
    """"Base class for all builder classes."""

    def __init__(self, num_classes=19):
        """Initializes a new instance of the BuilderBase class.

        Args:
            num_classes: The number of classes (=output channels).
        """
        self.num_classes = num_classes

    def add_conv(self,
                 network,
                 num_filters=64,
                 filter_size=(3, 3),
                 name="conv",
                 nonlinearity=True,
                 batch_norm=True,
                 bias=False):
        """Adds a convolution to the network stream.

        Args:
            network: The network stream.
            num_filters: The number of filters.
            filter_size: The filter size as tuple.
            name: The name of the block
            nonlinearity: True if a ReLU nonlinearity shall be added.
            batch_norm: True if a batch norm layer shall be added.
            bias: Whether or not to add a bias.

        Returns:
            The new network stream.
        """
        if bias:
            b = lasagne.init.Constant(0.0)
        else:
            b = None

        # Add the base convolutions
        network = lasagne.layers.Conv2DLayer(
            network,
            b=b,
            filter_size=filter_size,
            name=name + ".0",
            nonlinearity=lasagne.nonlinearities.linear,
            num_filters=num_filters,
            pad='same',
            W=lasagne.init.HeUniform('relu' if nonlinearity else 1.0))

        # Add the batch norm?
        if batch_norm:
            network = layers.BatchNormLayer(network, name=name + ".1")

        # Add the ReLU layer?
        if nonlinearity:
            network = lasagne.layers.NonlinearityLayer(
                network, nonlinearity=lasagne.nonlinearities.rectify)

        return network

    @staticmethod
    def log_softmax_4d(x):
        """4D log softmax function for dense classification tasks.

        Tensor layout: `(batch, classes, height, width)`. We normalize
        over the classes.

        Args:
            x: The input tenser.

        Returns:
            The output tensor.
        """
        try:
            x = theano.sandbox.cuda.basic_ops.gpu_contiguous(x)
            return theano.sandbox.cuda.dnn.GpuDnnSoftmax(
                tensor_format='bc01', algo="log", mode="channel")(x)
        except:
            x = theano.gpuarray.dnn.gpu_contiguous(x)
            return theano.gpuarray.dnn.GpuDnnSoftmax(
                algo="log", mode="channel")(x)


class FRRNBuilderBase(BuilderBase):
    """Base class for all FRRN builders."""

    def __init__(self, base_channels=32, lanes=32, multiplier=2, **kwargs):
        """Initializes a new instance of the FRRNBuilderBase.

        :param base_channels: The number of base_channels.
        :param lanes: The number of autobahn lanes.
        :param multiplier: The channel multiplier.
        """
        super(FRRNBuilderBase, self).__init__(**kwargs)

        self.base_channels = base_channels
        self.lanes = lanes
        self.multiplier = multiplier
        self.block_counter = 0
        self.module_counter = 0

    def get_module_name(self):
        """Returns the module name for the block-wise backpropagation algorithm.

        Returns:
            A new module name for the block-wise backpropagation algorithm.
        """
        name = "%04d.%04d" % (self.block_counter, self.module_counter)
        self.module_counter += 1
        return name

    def add_conv(self, **kwargs):
        """Adds a convolution to the network stream.

        Args:
            **kwargs: Arguments for `BuilderBase.add_conv`.

        Returns:
            New network stream with convolution attached.
        """
        kwargs["name"] = self.get_module_name()
        return super(FRRNBuilderBase, self).add_conv(**kwargs)

    def add_split(self, layers, nnet):
        """Adds a split to the network for the block-wise backprop algorithm.

        Args:
            layers The layers that define the cut.
            result  the nnet instance.
        """
        nnet.splits.append(layers)
        self.block_counter += 1
        self.module_counter = 0

    def add_frru(self,
                 pooling_stream,
                 residual_stream,
                 pooling,
                 multiplier=None):
        """Adds a full resolution residual unit.

        Args:
            pooling_stream: The main network stream (encoder/decode).
            residual_stream: The full-resolution residual stream.
            pooling: The pooling factor in the encoder/decoder hierarchy.
            multiplier: The multiplier that determines the number of channels.
                None -> use pooling

        Returns:
            `(pooling_stream, residual_stream)` The new pooling and residual
            streams.
        """
        # If we don't have a multiplier, use the pooling factor as multiplier.
        if multiplier is None:
            multiplier = pooling

        # Make sure that the number of channels is integer
        channels = int(self.base_channels * multiplier)

        # Store the initial residual input
        residual_input = residual_stream

        # If we work on a pooled image, we have to pool the residual stream as
        # well
        if pooling > 1:
            residual_input = lasagne.layers.MaxPool2DLayer(
                residual_input,
                stride=pooling,
                pool_size=(pooling, pooling))

        # Merge the two streams
        pooling_stream = lasagne.layers.ConcatLayer(
            [pooling_stream, residual_input])

        # Perform two convolutions on the concatenated features
        pooling_stream = self.add_conv(network=pooling_stream,
                                       num_filters=channels)
        pooling_stream = self.add_conv(network=pooling_stream,
                                       num_filters=channels)

        # Merge the result back into the residual stream
        residual = self.add_conv(
            network=pooling_stream,
            num_filters=self.lanes,
            filter_size=(1, 1),
            nonlinearity=False,
            batch_norm=False,
            bias=True)

        # If we work on a pooled image, we have to unpool the residual
        if pooling > 1:
            residual = lasagne.layers.Upscale2DLayer(residual,
                                                     scale_factor=pooling)

        residual_stream = lasagne.layers.ElemwiseSumLayer(
            [residual_stream, residual])

        return pooling_stream, residual_stream

    def add_ru(self, network, channels_in, channels_out):
        """Adds a residual unit consisting of two 3x3 convolutions.

        Args:
            network: The network stream.
            channels_in: The number of input channels.
            channels_out: The number of output channels.

        Returns:
            The network with a residual unit attached.
        """
        network_in = network

        # If the number of input channels is different from the number of output
        # channels, then we have to add a linear projection
        if channels_in != channels_out:
            network_in = self.add_conv(
                network=network_in,
                num_filters=channels_out,
                filter_size=(1, 1),
                batch_norm=False,
                nonlinearity=False,
                bias=False)

        network = self.add_conv(
            network=network, num_filters=channels_out)
        network = self.add_conv(
            network=network, num_filters=channels_out, nonlinearity=False)

        network = lasagne.layers.ElemwiseSumLayer([network, network_in])

        return network


class FRRNABuilder(FRRNBuilderBase):
    """Builds the FRRN A architecture."""

    def __init__(self, **kwargs):
        """Initializes a new instance of the FRRNABuilder.

        Args:
            **kwargs: Arguments for `FRRNBuilderBase.__ini__`.
        """
        super(FRRNABuilder, self).__init__(**kwargs)

    def build(self, input_var, input_shape):
        """Builds the network graph.

        Args:
            input_var: The input variable.
            input_shape: The input shape.

        Returns:
            The resulting network.
        """
        result = nnet.NeuralNetwork()

        pooling_stream = lasagne.layers.InputLayer(
            input_var=input_var,
            shape=input_shape)
        result.input_layers.append(pooling_stream)

        # First convolution
        pooling_stream = self.add_conv(
            network=pooling_stream,
            num_filters=self.base_channels,
            filter_size=(5, 5))

        # Add the full-res blocks
        pooling_stream = self.add_ru(pooling_stream,
                                     self.base_channels,
                                     self.base_channels)
        pooling_stream = self.add_ru(pooling_stream,
                                     self.base_channels,
                                     self.base_channels)
        pooling_stream = self.add_ru(pooling_stream,
                                     self.base_channels,
                                     self.base_channels)

        # Start the residual stream at a reduced channel count
        residual_stream = self.add_conv(network=pooling_stream,
                                        num_filters=self.lanes,
                                        filter_size=(1, 1),
                                        nonlinearity=False)

        self.add_split([pooling_stream, residual_stream], result)

        # Pooling stage / 2
        pooling_stream = lasagne.layers.MaxPool2DLayer(
            pooling_stream, stride=2, pool_size=(2, 2))
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 1)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 1)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 1)

        # Pooling stage / 4
        pooling_stream = lasagne.layers.MaxPool2DLayer(
            pooling_stream, stride=2, pool_size=(2, 2))
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 2)

        # Pooling stage / 8
        pooling_stream = lasagne.layers.MaxPool2DLayer(
            pooling_stream, stride=2, pool_size=(2, 2))
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            self.multiplier ** 3)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            self.multiplier ** 3)

        # Pooling stage / 16
        pooling_stream = lasagne.layers.MaxPool2DLayer(
            pooling_stream, stride=2, pool_size=(2, 2))
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 4,
            multiplier=self.multiplier ** 3)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 4,
            multiplier=self.multiplier ** 3)

        # Pooling stage / 8
        pooling_stream = layers.BilinearUpscaleLayer(pooling_stream, factor=2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 3,
            multiplier=self.multiplier ** 2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 3,
            multiplier=self.multiplier ** 2)

        # Pooling stage / 4
        pooling_stream = layers.BilinearUpscaleLayer(pooling_stream, factor=2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 2)

        # Pooling stage / 2
        pooling_stream = layers.BilinearUpscaleLayer(pooling_stream, factor=2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 1)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 1)

        # Pooling stage / 1 <-> Full resolution images
        pooling_stream = layers.BilinearUpscaleLayer(pooling_stream, factor=2)

        # Merge the two streams
        network = lasagne.layers.ConcatLayer(
            [pooling_stream, residual_stream])

        network = self.add_ru(
            network, self.base_channels + self.lanes, self.base_channels)
        network = self.add_ru(
            network, self.base_channels, self.base_channels)
        network = self.add_ru(
            network, self.base_channels, self.base_channels)

        # Classification layer
        network = self.add_conv(network=network,
                                num_filters=self.num_classes,
                                filter_size=(1, 1),
                                batch_norm=False,
                                nonlinearity=False,
                                bias=True)
        network = lasagne.layers.NonlinearityLayer(
            network, BuilderBase.log_softmax_4d)

        result.output_layers.append(network)

        return result


class FRRNBBuilder(FRRNBuilderBase):
    """Builds the FRRN B architecture."""

    def __init__(self, **kwargs):
        """Initializes a new instance of the FRRNABuilder.

        Args:
            **kwargs: Arguments for `FRRNBuilderBase.__ini__`.
        """
        super(FRRNBBuilder, self).__init__(**kwargs)

    def build(self, input_var, input_shape):
        """Builds the network graph.

        Args:
            input_var: The input variable.
            input_shape: The input shape.

        Returns:
            The resulting network.
        """
        result = nnet.NeuralNetwork()

        pooling_stream = lasagne.layers.InputLayer(
            input_var=input_var,
            shape=input_shape)
        result.input_layers.append(pooling_stream)

        # First convolution
        pooling_stream = self.add_conv(
            network=pooling_stream,
            num_filters=self.base_channels,
            filter_size=(5, 5))

        # Add the full-res blocks
        pooling_stream = self.add_ru(pooling_stream,
                                     self.base_channels,
                                     self.base_channels)
        pooling_stream = self.add_ru(pooling_stream,
                                     self.base_channels,
                                     self.base_channels)
        pooling_stream = self.add_ru(pooling_stream,
                                     self.base_channels,
                                     self.base_channels)

        # Start the residual stream at a reduced channel count
        residual_stream = self.add_conv(network=pooling_stream,
                                        num_filters=self.lanes,
                                        filter_size=(1, 1),
                                        nonlinearity=False)

        self.add_split([pooling_stream, residual_stream], result)

        # Pooling stage / 2
        pooling_stream = lasagne.layers.MaxPool2DLayer(
            pooling_stream, stride=2, pool_size=(2, 2))
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 1)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 1)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 1)

        self.add_split([pooling_stream, residual_stream], result)
        
        # Pooling stage / 4
        pooling_stream = lasagne.layers.MaxPool2DLayer(
            pooling_stream, stride=2, pool_size=(2, 2))
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 2)

        # Pooling stage / 8
        pooling_stream = lasagne.layers.MaxPool2DLayer(
            pooling_stream, stride=2, pool_size=(2, 2))
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            self.multiplier ** 3)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            self.multiplier ** 3)

        # Pooling stage / 16
        pooling_stream = lasagne.layers.MaxPool2DLayer(
            pooling_stream, stride=2, pool_size=(2, 2))
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 4,
            multiplier=self.multiplier ** 3)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 4,
            multiplier=self.multiplier ** 3)

        # Pooling stage / 32
        pooling_stream = lasagne.layers.MaxPool2DLayer(
            pooling_stream, stride=2, pool_size=(2, 2))
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 5,
            multiplier=self.multiplier ** 3)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 5,
            multiplier=self.multiplier ** 3)

        self.add_split([pooling_stream, residual_stream], result)

        # Pooling stage / 16
        pooling_stream = layers.BilinearUpscaleLayer(pooling_stream, factor=2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 4,
            multiplier=self.multiplier ** 2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 4,
            multiplier=self.multiplier ** 2)

        # Pooling stage / 8
        pooling_stream = layers.BilinearUpscaleLayer(pooling_stream, factor=2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 3,
            multiplier=self.multiplier ** 2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 3,
            multiplier=self.multiplier ** 2)

        # Pooling stage / 4
        pooling_stream = layers.BilinearUpscaleLayer(pooling_stream, factor=2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 2)

        # Pooling stage / 2
        pooling_stream = layers.BilinearUpscaleLayer(pooling_stream, factor=2)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 1)
        pooling_stream, residual_stream = self.add_frru(
            pooling_stream,
            residual_stream,
            pooling=self.multiplier ** 1)

        self.add_split([pooling_stream, residual_stream], result)
        # Pooling stage / 1 <-> Full resolution images
        pooling_stream = layers.BilinearUpscaleLayer(pooling_stream, factor=2)

        # Merge the two streams
        network = lasagne.layers.ConcatLayer(
            [pooling_stream, residual_stream])

        network = self.add_ru(
            network, self.base_channels + self.lanes, self.base_channels)
        network = self.add_ru(
            network, self.base_channels, self.base_channels)
        network = self.add_ru(
            network, self.base_channels, self.base_channels)

        # Classification layer
        network = self.add_conv(network=network,
                                num_filters=self.num_classes,
                                filter_size=(1, 1),
                                batch_norm=False,
                                nonlinearity=False,
                                bias=True)
        network = lasagne.layers.NonlinearityLayer(
            network, BuilderBase.log_softmax_4d)

        result.output_layers.append(network)

        return result
