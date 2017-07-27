"""Definitions for block-wise backpropagation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import re
import lasagne
import theano
import theano.tensor as T
from theano import subgraph_grad


def get_split_outputs(nnet, **kwargs):
    """Returns the predictions output and the outputs at the split layers.

    Args:
        nnet: The nnet instance that carries the network definition.
        **kwargs: Arguments for `lasagne.layers.get_output`.
    """
    all_outputs = lasagne.layers.get_output(
        nnet.output_layers + sum(nnet.splits, []), **kwargs)

    # Get the original predictions back
    predictions = all_outputs[:len(nnet.output_layers)]

    # Restore the network splits
    split_outputs = []
    split_shapes = []
    index = len(nnet.output_layers)
    for split in nnet.splits:
        split_outputs.append(all_outputs[index:index + len(split)])
        split_shapes.append([layer.output_shape for layer in split])
        index += len(split)

    return predictions, split_outputs, split_shapes


def compile_fwd_pass(network, input_vars):
    """Compiles and returns the theano forward pass function.

    During the forward pass, the tensor values at the split outputs are being
    cached and the batch norm statistics are being updated.

    Args:
        network: The network instance.
        input_vars: A list of input variables.

    Returns:
    A tuple consisting of the forward pass function and a list of theano
    variables that cache the tensor values at the shared outputs.
    """
    split_outputs = get_split_outputs(network)
    all_predictions, split_outputs, split_shapes = split_outputs

    updates = collections.OrderedDict()
    split_values = []

    for outputs, shapes in zip(split_outputs, split_shapes):
        for output, shape in zip(outputs, shapes):
            # Create a new shared variable for the output
            var = theano.shared(np.empty(shape, dtype='float32'))
            updates[var] = output
            split_values.append(var)

    fwd_fn = theano.function(
        inputs=input_vars,
        outputs=all_predictions,
        updates=updates,
        on_unused_input='ignore'
    )

    return fwd_fn, split_values


def compile_bwd_pass(network, loss_fn, split_values, input_vars):
    """Compiles the backward pass.

    Args:
        network: The network instance.
        loss_fn: A function that receives the network's predictions and returns
            a theano scalar loss.
        split_values: The values at the split outputs.
        input_vars: The graph inputs.

    Returns:
    A function that computes and returns the gradients for all parameters.
    """
    # Construct a new graph where we don't update the batch norm stats for
    # every run
    split_outputs = get_split_outputs(network,
                                      batch_norm_update_averages=False)

    # Get the parameter tensors split by block
    param_blocks, _ = split_params(network)
    all_predictions, split_outputs, split_shapes = split_outputs

    # Map the cached values from the fwd pass to the given split outputs
    given_values = collections.OrderedDict()
    i = 0
    for outputs, shapes in zip(split_outputs, split_shapes):
        for output, _ in zip(outputs, shapes):
            given_values[output] = split_values[i]
            i += 1

    grad_fns = compile_grad_functions(
        split_outputs,
        param_blocks,
        input_vars,
        loss_fn(*all_predictions),
        given_values)
    return lambda *args: compute_grads(grad_fns, param_blocks, *args)


def compile_gd_step(network, loss_fn, input_vars, update_fn):
    """Compiles the backward pass.

    Args:
        network: The network instance.
        loss_fn: A function that receives the network's predictions and returns
            a theano scalar loss.
        split_values: The values at the split outputs.
        input_vars: The graph inputs.
        update_fn: A function that computes the gradient descent update.

    Returns:
    A function that computes and returns the gradients for all parameters.
    """
    # Construct a new graph where we don't update the batch norm stats for
    # every run.
    bn_updates = collections.OrderedDict()
    split_outputs = get_split_outputs(network,
                                      batch_norm_update_averages=bn_updates)

    # Get the parameter tensors split by block
    param_blocks, _ = split_params(network)
    all_predictions, split_outputs, split_shapes = split_outputs

    # Create the update ops for storing the intermediate split outputs. This
    # avoids too many forward passes.
    split_updates = collections.OrderedDict()
    given_values = collections.OrderedDict()
    split_values = []

    for outputs, shapes in zip(split_outputs, split_shapes):
        for output, shape in zip(outputs, shapes):
            # Create a new shared variable for the output
            var = theano.shared(np.empty(shape, dtype='float32'))
            split_updates[var] = output
            given_values[output] = var
            split_values.append(var)

    grad_fns = compile_grad_descent_functions(
        bn_updates,
        split_updates,
        split_outputs,
        param_blocks,
        input_vars,
        loss_fn(*all_predictions),
        given_values,
        update_fn)
    return lambda *args: grad_descent_step(grad_fns, *args)


def _like(tensor):
    """Returns a tensor type from a tensor.

    Args:
        tensor: The theano tensor.

    Returns:
        A new tensor type matching x's type.
    """
    return T.TensorType('float32', tensor.broadcastable)()


def split_params(nnet):
    """Splits the parameters of the network into blocks.

    Args:
        nnet: The nnet instance that carries the network definition.

    Returns:
        The parameters for each block and the parameters for the entire network.
    """
    params = nnet.get_params(trainable=True)
    pattern = re.compile(r"([0-9]+\.)+")
    params = sorted(params, key=lambda x: pattern.match(x.name).group())

    # Get the parameters that are associated with the split
    param_blocks = []
    for i in range(len(nnet.splits) + 1):
        block = [pattern for pattern in params
                 if pattern.name.startswith('%04d.' % i)]
        param_blocks.append(block)

    return param_blocks, params


def compile_grad_functions(split_outputs,
                           param_blocks,
                           input_vars,
                           loss,
                           givens):
    """Compiles functions that compute the gradients for each block.

    Args:
        split_outputs: The split nodes.
        param_blocks: The parameters for each block.
        input_vars: The input variables to the network.
        loss: The training loss.
        givens: A dictionary of given variable values (computed during a
            dedicated forward pass).
    """
    grad_fns = []
    for i in range(len(param_blocks) - 1, -1, -1):
        if i > 0:
            end = split_outputs[i - 1]
        else:
            end = []

        if i < len(split_outputs):
            # Create gradient variables for all split vars
            start = collections.OrderedDict()
            for s in split_outputs[i]:
                start[s] = _like(s)
            start_vars = list(start.values())
        else:
            start = None
            start_vars = []

        if start is None:
            grads, out_grads = subgraph_grad(
                end=end,
                cost=loss,
                wrt=param_blocks[i]
            )
            # Create the grad function
            grad_fns.append(theano.function(
                inputs=input_vars + start_vars,
                outputs=[loss] + grads + out_grads,
                on_unused_input='ignore'
            ))
        else:
            grads, out_grads = subgraph_grad(
                start=start,
                end=end,
                wrt=param_blocks[i]
            )
            # Create the grad function
            grad_fns.append(theano.function(
                inputs=input_vars + start_vars,
                outputs=grads + out_grads,
                on_unused_input='ignore',
                givens=givens
            ))

    return grad_fns[::-1]


def compile_grad_descent_functions(bn_updates,
                                   split_updates,
                                   split_outputs,
                                   param_blocks,
                                   input_vars,
                                   loss,
                                   givens,
                                   update_fn):
    """Compiles functions that perform a gradient descent stop for each block.

    This function is complementary to `compile_grad_functions`.

    Args:
        bn_updates: A dictionary for updating the BN statistics.
        split_updates: A dictionary containing the update ops for the
            intermediate split outputs.
        split_outputs: The split nodes.
        param_blocks: The parameters for each block.
        input_vars: The input variables to the network.
        loss: The training loss.
        givens: A dictionary of given variable values (computed during a
            dedicated forward pass).
        update_fn: A lasagne update function that takes a list of gradient and
            parameters and returns a theano update dict.
    """
    grad_fns = []
    for i in range(len(param_blocks) - 1, -1, -1):
        if i > 0:
            end = split_outputs[i - 1]
        else:
            end = []

        if i < len(split_outputs):
            # Create gradient variables for all split vars
            start = collections.OrderedDict()
            for s in split_outputs[i]:
                start[s] = _like(s)
            start_vars = list(start.values())
        else:
            start = None
            start_vars = []

        if start is None:
            grads, out_grads = subgraph_grad(
                end=end,
                cost=loss,
                wrt=param_blocks[i],
            )

            # Compute the gradient descent update
            updates = update_fn(loss_or_grads=grads, params=param_blocks[i])

            # Update the BN statistics and store the intermediate outputs.
            updates.update(bn_updates)
            updates.update(split_updates)

            # Create the grad function
            grad_fns.append(theano.function(
                inputs=input_vars + start_vars,
                outputs=[loss] + out_grads,
                updates=updates,
                on_unused_input='ignore'
            ))
        else:
            grads, out_grads = subgraph_grad(
                start=start,
                end=end,
                wrt=param_blocks[i]
            )

            # Compute the gradient descent update
            updates = update_fn(loss_or_grads=grads, params=param_blocks[i])

            # Create the grad function
            grad_fns.append(theano.function(
                inputs=input_vars + start_vars,
                outputs=out_grads,
                updates=updates,
                on_unused_input='ignore',
                givens=givens
            ))

    return grad_fns


def compute_grads(grad_fns, param_blocks, *args):
    """Computes the gradients block wise.

    Args:
        grad_fns: The gradient computation functions.
        param_blocks: The parameters for each block.
        *args: The values of the input variables. (e.g. images and targets).
    """
    args = list(args)
    num_grads = len(grad_fns)
    acc_grads = [None] * num_grads
    prev = []

    # Compute the first iteration
    i = num_grads - 1
    result = grad_fns[i](*(args + prev))
    loss = result[0]
    result = result[1:]

    current = result[:len(param_blocks[i])]
    prev = result[len(param_blocks[i]):]
    acc_grads[len(grad_fns) - 1 - i] = current[::-1]

    for i in range(num_grads - 2, -1, -1):
        result = grad_fns[i](*(args + prev))

        current = result[:len(param_blocks[i])]
        prev = result[len(param_blocks[i]):]
        acc_grads[num_grads - 1 - i] = current[::-1]

    return loss, sum(acc_grads, [])[::-1]


def grad_descent_step(grad_fns, *args):
    """Computes the gradients block wise.

    Args:
        grad_fns: The gradient computation functions.
        *args: The values of the input variables. (e.g. images and targets).
    """
    args = list(args)

    # Compute the first iteration
    result = grad_fns[0](*args)
    loss = result[0]
    prev = result[1:]

    # Compute the remaining iterations
    for i in range(1, len(grad_fns)):
        prev = grad_fns[i](*(args + prev))

    return loss


def get_gradient_variables(params):
    """Creates a new tensor for each input tensor.

    Args:
        params: A list of tensors.

    Returns:
        A list of zero-initialized tensors of the same shapes.
    """
    return [T.zeros_like(p) for p in params]
