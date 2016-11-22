import theano.tensor as T
from lasagne.layers import Layer, MergeLayer


class BilinearUpscaleLayer(Layer):
    """
    This layer upscales the 4D input tensor along the trailing spatial dimensions using bilinear interpolation.
    You have to specify image dimensions in order to use this layer - even if you want to have a fully convolutional
    network.
    """
    def __init__(self, incoming, factor, **kwargs):
        """
        Initializes a new instance of the BilinearUpscaleLayer class.

        :param incoming: The incoming network stream
        :param factor: The factor by which to upscale the input
        """
        super(BilinearUpscaleLayer, self).__init__(incoming, **kwargs)
        self.factor = factor

    def get_output_shape_for(self, input_shape):
        """
        Computes the output shape of the layer given the input shape.

        :param input_shape: The input shape
        :return: The output shape
        """
        return input_shape[0], input_shape[1], self.factor * input_shape[2], self.factor * input_shape[3]

    def get_output_for(self, input, **kwargs):
        """
        Constructs the Theano graph for this layer
        :param input: Symbolic input variable
        :return: Symbolic output variable
        """
        return T.nnet.abstract_conv.bilinear_upsampling(input, self.factor)
