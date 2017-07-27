"""Tests the hybrid training module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from dltools import architectures, nnet, hybrid_training, layers
import functools
import numpy as np
import theano
import theano.tensor as T
import lasagne
import unittest


class DummyBuilder(architectures.FRRNBuilderBase):
    """Dummy network builder."""

    def __init__(self):
        """Initializes a new instance of the DummyBuilder class."""
        super(DummyBuilder, self).__init__()
        self.bn_layers = []

    def add_dense_unit(self, network, num_units):
        """Adds a dense layer with batch norm."""
        network = lasagne.layers.DenseLayer(
            network, num_units=num_units, b=None, name=self.get_module_name())
        network = layers.BatchNormLayer(network, name=self.get_module_name())
        self.bn_layers.append(network)
        network = lasagne.layers.NonlinearityLayer(
            network, nonlinearity=lasagne.nonlinearities.tanh)
        return network

    def build(self, input_var, input_shape):
        """Builds the network graph.

        Args:
            input_var: The input variable.
            input_shape: The input shape.

        Returns:
            The resulting network.
        """
        result = nnet.NeuralNetwork()

        network = lasagne.layers.InputLayer(
            input_var=input_var,
            shape=input_shape)
        result.input_layers.append(network)

        network = self.add_dense_unit(network, 32)
        network = self.add_dense_unit(network, 32)
        network = self.add_dense_unit(network, 32)

        self.add_split([network], result)

        network = self.add_dense_unit(network, 32)
        network = self.add_dense_unit(network, 32)
        network = self.add_dense_unit(network, 32)

        self.add_split([network], result)

        network = self.add_dense_unit(network, 32)
        network = self.add_dense_unit(network, 32)
        network = self.add_dense_unit(network, 32)

        self.add_split([network], result)

        network = self.add_dense_unit(network, 32)
        network = self.add_dense_unit(network, 32)
        network = self.add_dense_unit(network, 32)

        network = lasagne.layers.DenseLayer(
            network, num_units=1, nonlinearity=None, name=self.get_module_name()
        )

        result.output_layers.append(network)

        return result


TestNetwork = collections.namedtuple("TestNetwork", [
    "givens", "given_values", "builder", "network", "loss_fn", "fwd_pass",
    "bwd_pass", "gd_step"
])


class TestHybridTraining(unittest.TestCase):
    """Tests the hybrid training module."""

    def _create_network_instance(self):
        """Creates a new instance of TestNetwork."""
        input_var = T.fmatrix()

        input_shape = (10, 10)
        target_var = T.fvector()

        np.random.seed(0)
        input_val = np.random.normal(0, 1, input_shape).astype('float32')
        target_val = 10 * np.ones(input_shape[:1], dtype='float32')
        givens = {
            input_var: input_val,
            target_var: target_val
        }
        given_values = [input_val, target_val]

        builder = DummyBuilder()
        network = builder.build(input_var, input_shape)

        loss_fn = lambda p: T.mean(T.sqr(p[:, 0] - target_var))

        update_fn = functools.partial(lasagne.updates.sgd, learning_rate=1)

        fwd_pass, split_values = hybrid_training.compile_fwd_pass(
            network, [input_var])
        bwd_pass = hybrid_training.compile_bwd_pass(
            network, loss_fn, split_values, [input_var, target_var])
        gd_step = hybrid_training.compile_gd_step(
            network, loss_fn, [input_var, target_var], update_fn)

        return TestNetwork(
            givens=givens,
            given_values=given_values,
            builder=builder,
            network=network,
            loss_fn=loss_fn,
            fwd_pass=fwd_pass,
            bwd_pass=bwd_pass,
            gd_step=gd_step
        )

    def setUp(self):
        """Sets up the test case."""
        self.network0 = self._create_network_instance()
        self.network1 = self._create_network_instance()

    def test_gradients(self):
        """Tests if the gradients of the block backprop are correct."""
        # Arrange
        # Compile the normal gradient function
        _, params = hybrid_training.split_params(self.network0.network)
        predictions = lasagne.layers.get_output(
            self.network0.network.output_layers)[0]
        loss = self.network0.loss_fn(predictions)

        grad_fn = theano.function(
            inputs=[],
            outputs=[loss] + T.grad(loss, params),
            givens=self.network0.givens
        )

        # Act
        exp_values = grad_fn()
        self.network0.fwd_pass(self.network0.given_values[0])
        real_values = self.network0.bwd_pass(*self.network0.given_values)

        # Assert
        for e, v in zip(exp_values[1:], real_values[1]):
            np.testing.assert_allclose(v, e, rtol=1e-6, atol=1e-6)

    def test_loss(self):
        """Tests if the loss is computed correctly."""
        # Arrange
        # Compile the normal gradient function
        _, params = hybrid_training.split_params(self.network0.network)
        predictions = lasagne.layers.get_output(
            self.network0.network.output_layers)[0]
        loss = self.network0.loss_fn(predictions)

        loss_fn = theano.function(
            inputs=[],
            outputs=loss,
            givens=self.network0.givens
        )

        # Act
        exp_loss = loss_fn()

        self.network0.fwd_pass(self.network0.given_values[0])
        real_loss, _ = self.network0.bwd_pass(*self.network0.given_values)

        # Assert
        self.assertEqual(exp_loss, real_loss)

    def test_loss_grad_descent(self):
        """Tests if the loss is computed correctly in a GD step."""
        # Arrange
        # Compile the normal gradient function
        _, params = hybrid_training.split_params(self.network0.network)
        predictions = lasagne.layers.get_output(
            self.network0.network.output_layers)[0]
        loss = self.network0.loss_fn(predictions)

        loss_fn = theano.function(
            inputs=[],
            outputs=loss,
            givens=self.network0.givens
        )

        # Act
        exp_loss = loss_fn()

        self.network0.fwd_pass(self.network0.given_values[0])
        real_loss = self.network0.gd_step(*self.network0.given_values)

        # Assert
        self.assertEqual(exp_loss, real_loss)

    def test_grad_descent(self):
        """Tests if the loss is computed correctly in a GD step."""
        # Arrange
        # Create the update function
        _, params = hybrid_training.split_params(self.network0.network)
        grad_vars = hybrid_training.get_gradient_variables(params)

        # Choose whatever optimizer you like
        updates = lasagne.updates.sgd(
            grad_vars, params, learning_rate=1)

        update_fn = theano.function(
            inputs=grad_vars,
            updates=updates,
        )

        # Act
        self.network0.fwd_pass(self.network0.given_values[0])
        _, grads = self.network0.bwd_pass(*self.network0.given_values)
        update_fn(*grads)

        self.network1.gd_step(*self.network1.given_values)

        weights0 = lasagne.layers.get_all_param_values(
            self.network0.network.output_layers)
        weights1 = lasagne.layers.get_all_param_values(
            self.network1.network.output_layers)

        # Assert
        for w0, w1 in zip(weights0, weights1):
            np.testing.assert_allclose(w0, w1, rtol=1e-6, atol=1e-6)

    def test_batch_norm(self):
        """Tests if batch norm statistics are aggregated correctly."""
        # Arrange
        # Act
        # Record the value of the batch norm statistics for all bn layers
        bn_means = [layer.mean.get_value()
                    for layer in self.network0.builder.bn_layers]
        bn_inv_stds = [layer.inv_std.get_value() for layer in
                       self.network0.builder.bn_layers]

        self.network0.fwd_pass(self.network0.given_values[0])

        # Record the values again
        bn_means_fwd_pass = [layer.mean.get_value() for layer in
                             self.network0.builder.bn_layers]
        bn_inv_stds_fwd_pass = [layer.inv_std.get_value() for layer in
                                self.network0.builder.bn_layers]

        self.network0.bwd_pass(*self.network0.given_values)

        bn_means_bwd_pass = [layer.mean.get_value() for layer in
                             self.network0.builder.bn_layers]
        bn_inv_stds_bwd_pass = [layer.inv_std.get_value() for layer in
                                self.network0.builder.bn_layers]
        # Assert
        # The initial value must differ from the value after the forward pass
        # because the statistics should have been updated
        vars_0 = bn_means + bn_inv_stds
        vars_1 = bn_means_fwd_pass + bn_inv_stds_fwd_pass
        vars_2 = bn_means_bwd_pass + bn_inv_stds_bwd_pass

        for e, v in zip(vars_0, vars_1):
            diff = np.max(np.abs(e - v))
            self.assertGreater(diff, 1e-5)

        for e, v in zip(vars_1, vars_2):
            np.testing.assert_equal(v, e)

    def test_batch_norm_gd(self):
        """Tests if batch norm statistics are aggregated correctly."""
        # Arrange
        # Act
        # Record the value of the batch norm statistics for all bn layers
        bn_means = [layer.mean.get_value()
                    for layer in self.network0.builder.bn_layers]
        bn_inv_stds = [layer.inv_std.get_value() for layer in
                       self.network0.builder.bn_layers]
        self.network0.gd_step(*self.network0.given_values)

        bn_means_bwd_pass = [layer.mean.get_value() for layer in
                             self.network0.builder.bn_layers]
        bn_inv_stds_bwd_pass = [layer.inv_std.get_value() for layer in
                                self.network0.builder.bn_layers]
        # Assert
        # The initial value must differ from the value after the forward pass
        # because the statistics should have been updated
        vars_0 = bn_means + bn_inv_stds
        vars_1 = bn_means_bwd_pass + bn_inv_stds_bwd_pass

        for e, v in zip(vars_0, vars_1):
            diff = np.max(np.abs(e - v))
            self.assertGreater(diff, 1e-5)


if __name__ == '__main__':
    unittest.main()
