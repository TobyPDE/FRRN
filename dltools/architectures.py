from dltools.layers import BilinearUpscaleLayer
from . import nnet

import lasagne
import theano
import theano.tensor as T
import math


class AbstractBuilder(object):
    """
    This is the base class for all network builders.
    """

    def __init__(self, num_classes=19, use_virtual_bn=False):
        """
        Initializes a new instance of the AbstractBuilder class.
        :param num_classes: The number of classes (=output channels).
        :param use_virtual_bn: If true, virtual batch normaliaztion layers are used for virtual batch training.
        """
        self.num_classes = num_classes
        self.use_virtual_bn = use_virtual_bn
        self.alpha = 0.1

    def add_conv(self, network, nnet, num_filters=64, filter_size=(5, 5), name="conv", nonlinearity=True, bn=True,
                 bias=False, stride=1, zero_init=False):
        """
        Adds a convolution to the network stream

        :param network: The network stream.
        :param nnet: The neural network instance.
        :param num_filters: The number of filters.
        :param filter_size: The filter size as tuple.
        :param name: The name of the block
        :param nonlinearity: True if a ReLU nonlinearity shall be added.
        :param bn: True if a batch norm layer shall be added
        :return: The new network stream
        """
        if bias:
            b = lasagne.init.Constant(0.0)
        else:
            b = None

        # Add the base convolutions
        network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=num_filters,
            filter_size=filter_size,
            pad='same',
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.HeUniform('relu' if nonlinearity else 1.0) if not zero_init else lasagne.init.Constant(0.0),
            b=b,
            name=name + ".0",
            stride=(stride, stride))

        # Add the batch norm?
        if bn:
            network = lasagne.layers.BatchNormLayer(network, name=name + ".1", alpha=self.alpha)
            nnet.batchnorm_layers.append(network)

        # Add the ReLU layer?
        if nonlinearity:
            network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify)

        return network

    @staticmethod
    def accurate_softmax_4d(x):
        """
        Softmax function in 4d. We assume the following layout: (batch, classes, height, width). Hence, we normalize
        over the classes.

        :param x: The input tenser
        :return: The output tensor (normalized)
        """
        x = theano.sandbox.cuda.basic_ops.gpu_contiguous(x)
        return theano.sandbox.cuda.dnn.GpuDnnSoftmax(tensor_format='bc01', algo="accurate", mode="channel")(x)

    @staticmethod
    def log_softmax_4d(x):
        """
        Softmax function in 4d. We assume the following layout: (batch, classes, height, width). Hence, we normalize
        over the classes.

        :param x: The input tenser
        :return: The output tensor (normalized)
        """
        x = theano.sandbox.cuda.basic_ops.gpu_contiguous(x)
        return theano.sandbox.cuda.dnn.GpuDnnSoftmax(tensor_format='bc01', algo="log", mode="channel")(x)


class AbstractFRRNBuilder(AbstractBuilder):
    """
    Base class for all FRRN builders.
    """

    def __init__(self, base_channels=32, lanes=32, multiplier=2, **kwargs):
        """
        Initializes a new instance of the Autobahnnet2Builder.

        :param base_channels: The number of base_channels.
        :param lanes: The number of autobahn lanes.
        :param multiplier: The channel multiplier.
        """
        super().__init__(**kwargs)

        self.base_channels = base_channels
        self.lanes = lanes
        self.multiplier = multiplier
        self.block_counter = 0
        self.module_counter = 0
        self.alpha = 1 - math.pow(1 - 0.1, 1 / 1)

    def get_module_name(self):
        """
        Returns the module name for the block-wise backpropagation algorithm.
        Returns
        -------
        A new module name for the blockwise backpropagation algorithm.
        """
        name = "%04d.%04d" % (self.block_counter, self.module_counter)
        self.module_counter += 1
        return name

    def add_conv(self, network, nnet, num_filters=64, filter_size=(5, 5), name="conv", nonlinearity=True, bn=True,
                 bias=False, stride=1, zero_init=False):
        """
        Adds a convolution with the fitting module name for the block-wise backprop algorithm.
        Parameters
        ----------
        network The lasagne network stream
        nnet The nnet instance
        num_filters The number of filters
        filter_size the filter size
        name The name of the convolution. Use "conv" for the block-wise backprop algorithm.
        nonlinearity Whether or not a non-linearity shall be used.
        bn Whether of not a batch norm layer shall be used
        bias Whether or not a bias term shall be used
        stride the stride of the convolution

        Returns
        -------
        The new lasagne network stream.
        """
        if name == "conv":
            name = self.get_module_name()
        return super(AbstractFRRNBuilder, self).add_conv(
            network,
            nnet,
            num_filters=num_filters,
            filter_size=filter_size,
            name=name,
            nonlinearity=nonlinearity,
            bn=bn,
            bias=bias,
            zero_init=zero_init)

    def add_split(self, layers, nnet):
        """
        Adds a split/cut to the network for the block-wise backprop algorithm.
        Parameters
        ----------
        layers The layers that define the cut
        result  the nnet instance
        """
        nnet.splits.append(layers)
        self.block_counter += 1
        self.module_counter = 0

    def add_frru(self, encoder_decoder_stream, full_res_stream, nnet, pooling, multiplier=None):
        """
        Adds a full resolution residual unit.

        :param encoder_decoder_stream: The main network stream (encoder/decode).
        :param full_res_stream: The autobahn stream (full resolution).
        :param nnet: The nnet class.
        :param pooling: The pooling factor in the encoder/decoder hierarchy.
        :param multiplier: The multiplier that determines the number of channels. None -> use pooling
        :return: network stream, autobahn stream.
        """

        # If we don't have a multiplier, use the pooling factor.
        if multiplier is None:
            multiplier = pooling

        # Make sure that the number of channels is integer
        channels = int(self.base_channels * multiplier)
        # Store the initial autobahn input
        autobahn_input = full_res_stream

        # If we work on a pooled image, we have to pool the autobahn as well
        if pooling > 1:
            autobahn_input = lasagne.layers.MaxPool2DLayer(autobahn_input, stride=pooling, pool_size=(pooling, pooling))

        # Merge the streams
        encoder_decoder_stream = lasagne.layers.ConcatLayer([encoder_decoder_stream, autobahn_input])

        # Perform two convolutions
        encoder_decoder_stream = self.add_conv(encoder_decoder_stream, nnet, channels, (3, 3))
        encoder_decoder_stream = self.add_conv(encoder_decoder_stream, nnet, channels, (3, 3))

        # Merge the result back into the autobahn
        autobahn_output = self.add_conv(
            encoder_decoder_stream,
            nnet,
            self.lanes,
            (1, 1),
            nonlinearity=False,
            bn=False,
            bias=True)

        # If we work on a pooled image, we have to unpool the result again
        if pooling > 1:
            autobahn_output = lasagne.layers.Upscale2DLayer(autobahn_output, scale_factor=pooling)

        full_res_stream = lasagne.layers.ElemwiseSumLayer([full_res_stream, autobahn_output])

        return encoder_decoder_stream, full_res_stream

    def add_ru(self, network, nnet, channels_in, channels_out):
        """
        Adds a residual unit consisting of two 3x3 convolutions.

        :param network: The network stream.
        :param nnet: The nnet instance.
        :param channels_out: The number of output channels
        :return: The network stream with a residual block attached.
        """
        network_in = network

        # If the number of input channels is different from the number of output channels, then we have to
        # add a linear projection
        if channels_in != channels_out:
            network_in = self.add_conv(network_in, nnet, channels_out, (1, 1), bn=False, nonlinearity=False, bias=False)

        network = self.add_conv(network, nnet, channels_out, (3, 3))
        network = self.add_conv(network, nnet, channels_out, (3, 3), nonlinearity=False)

        network = lasagne.layers.ElemwiseSumLayer([network, network_in])

        return network


class FRRNBBuilder(AbstractFRRNBuilder):
    """
    Builds the FRRN B architecture.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new instance of the FRRNBBuilder.

        :param base_channels: The number of base_channels.
        :param lanes: The number of autobahn lanes.
        :param multiplier: The channel multiplier.
        """
        super(FRRNBBuilder, self).__init__(**kwargs)
        self.alpha = 1 - math.pow(1 - 0.1, 1 / 5)

    def build(self, input_var, input_shape):
        """
        Builds the network.

        :param input_var: The input variable.
        :param input_shape: The input shape.
        :return: The resulting network.
        """
        result = nnet.NeuralNetwork()

        network = lasagne.layers.InputLayer(
            input_var=input_var,
            shape=input_shape)
        result.input_layers.append(network)

        network = self.add_conv(network, result, self.base_channels, (5, 5))

        # Add the full-res block
        network = self.add_ru(network, result, self.base_channels, self.base_channels)
        network = self.add_ru(network, result, self.base_channels, self.base_channels)
        network = self.add_ru(network, result, self.base_channels, self.base_channels)

        autobahn = self.add_conv(network, result, self.lanes, (1, 1), nonlinearity=False)
        
        self.add_split([network, autobahn], result)
        self.alpha = 1 - math.pow(1 - 0.1, 1 / 4)
        
        # Pooling
        network = lasagne.layers.MaxPool2DLayer(network, stride=2, pool_size=(2, 2))
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 1)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 1)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 1)
        
        self.add_split([network, autobahn], result)
        self.alpha = 1 - math.pow(1 - 0.1, 1 / 3)
        
        # Pooling
        network = lasagne.layers.MaxPool2DLayer(network, stride=2, pool_size=(2, 2))
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 2)

        # Pooling
        network = lasagne.layers.MaxPool2DLayer(network, stride=2, pool_size=(2, 2))
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 3)
        
        self.add_split([network, autobahn], result)
        self.alpha = 1 - math.pow(1 - 0.1, 1 / 2)

        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 3)

        # Pooling
        network = lasagne.layers.MaxPool2DLayer(network, stride=2, pool_size=(2, 2))

        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 4, multiplier=self.multiplier ** 3)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 4, multiplier=self.multiplier ** 3)

        network = lasagne.layers.MaxPool2DLayer(network, stride=2, pool_size=(2, 2))
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 5, multiplier=self.multiplier ** 3)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 5, multiplier=self.multiplier ** 3)

        # Unpooling
        network = BilinearUpscaleLayer(network, factor=2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 4, multiplier=self.multiplier ** 2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 4, multiplier=self.multiplier ** 2)

        # Unpooling
        network = BilinearUpscaleLayer(network, factor=2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 3, multiplier=self.multiplier ** 2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 3, multiplier=self.multiplier ** 2)

        # Unpooling
        network = BilinearUpscaleLayer(network, factor=2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 2)
        
        # Unpooling
        network = BilinearUpscaleLayer(network, factor=2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 1)
        
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 1)

        self.add_split([network, autobahn], result)
        self.alpha = 1 - math.pow(1 - 0.1, 1 / 1)
        
        # Unpooling
        network = BilinearUpscaleLayer(network, factor=2)
        network = lasagne.layers.ConcatLayer([network, autobahn])

        network = self.add_ru(network, result, self.base_channels + self.lanes, self.base_channels)
        network = self.add_ru(network, result, self.base_channels, self.base_channels)
        network = self.add_ru(network, result, self.base_channels, self.base_channels)
        
        # Classification layer
        network = self.add_conv(network, result, self.num_classes, (1, 1), bn=False, nonlinearity=False, bias=True)
        network = lasagne.layers.NonlinearityLayer(network, AbstractBuilder.log_softmax_4d)

        result.output_layers.append(network)

        return result


class FRRNCBuilder(AbstractFRRNBuilder):
    """
    Builds the FRRNCBuilder architecture. This is a smaller architecture intended for fast experimentation. Train
    on four times subsampled images.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new instance of the FRRNCBuilder.

        :param base_channels: The number of base_channels.
        :param lanes: The number of autobahn lanes.
        :param multiplier: The channel multiplier.
        """
        super(FRRNCBuilder, self).__init__(**kwargs)

    def build(self, input_var, input_shape):
        """
        Builds the network.

        :param input_var: The input variable.
        :param input_shape: The input shape.
        :return: The resulting network.
        """
        result = nnet.NeuralNetwork()

        network = lasagne.layers.InputLayer(
            input_var=input_var,
            shape=input_shape)
        result.input_layers.append(network)

        network = self.add_conv(network, result, self.base_channels, (5, 5))

        # Add the full-res block
        network = self.add_ru(network, result, self.base_channels, self.base_channels)
        network = self.add_ru(network, result, self.base_channels, self.base_channels)
        network = self.add_ru(network, result, self.base_channels, self.base_channels)

        autobahn = self.add_conv(network, result, self.lanes, (1, 1), nonlinearity=False)

        # Pooling
        network = lasagne.layers.MaxPool2DLayer(network, stride=2, pool_size=(2, 2))
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 1)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 1)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 1)

        # Pooling
        network = lasagne.layers.MaxPool2DLayer(network, stride=2, pool_size=(2, 2))
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 2)

        # Pooling
        network = lasagne.layers.MaxPool2DLayer(network, stride=2, pool_size=(2, 2))
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 3)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 3)

        # Unpooling
        network = BilinearUpscaleLayer(network, factor=2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 2)

        # Unpooling
        network = BilinearUpscaleLayer(network, factor=2)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 1)
        network, autobahn = self.add_frru(network, autobahn, result, self.multiplier ** 1)

        # Unpooling
        network = BilinearUpscaleLayer(network, factor=2)
        network = lasagne.layers.ConcatLayer([network, autobahn])

        network = self.add_ru(network, result, self.base_channels + self.lanes, self.base_channels)
        network = self.add_ru(network, result, self.base_channels, self.base_channels)
        network = self.add_ru(network, result, self.base_channels, self.base_channels)

        # Classification layer
        network = self.add_conv(network, result, self.num_classes, (1, 1), bn=False, nonlinearity=False, bias=True)
        network = lasagne.layers.NonlinearityLayer(network, AbstractBuilder.log_softmax_4d)

        result.output_layers.append(network)

        return result
