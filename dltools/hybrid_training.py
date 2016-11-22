from collections import OrderedDict

import re
import lasagne
import theano
import theano.tensor as T
from theano import subgraph_grad


def like(x):
    return T.TensorType('float32', x.broadcastable)()


def get_split_outputs(network, **kwargs):
    """
    Returns the predictions output and the outputs at the split layers.
    """
    all_outputs = lasagne.layers.get_output(network.output_layers + sum(network.splits, []), **kwargs)

    # Get the original predictions back
    predictions = all_outputs[:len(network.output_layers)]

    # Restore the network splits
    split_outputs = []
    index = len(network.output_layers)
    for s in network.splits:
        split_outputs.append(all_outputs[index:index + len(s)])
        index += len(s)

    return predictions, split_outputs


def split_params(network):
    """
    Splits the parameters of the network into blocks.
    """
    params = network.get_params(trainable=True)
    p = re.compile("([0-9]+\.)+")
    params = sorted(params, key=lambda x: p.match(x.name).group())

    # Get the parameters that are associated with the split
    param_blocks = []
    for i in range(len(network.splits) + 1):
        block = [p for p in params if p.name.startswith('%04d.' % i)]
        param_blocks.append(block)

    return param_blocks, params


def compile_grad_functions(split_outputs, param_blocks, input_vars, loss, givens):
    """
    Compiles functions that compute the gradients for each block given the preceding block.
    :return:
    """
    grad_fns = []
    for i in range(len(param_blocks) - 1, -1, -1):
        if i > 0:
            end = split_outputs[i - 1]
        else:
            end = []

        if i < len(split_outputs):
            # Create gradient variables for all split vars
            start = OrderedDict()
            for s in split_outputs[i]:
                start[s] = like(s)
            start_vars = list(start.values())
        else:
            start = None
            start_vars = []

        if start is None:
            grads, next = subgraph_grad(
                start=start,
                end=end,
                cost=loss,
                wrt=param_blocks[i]
            )
            # Create the grad function
            grad_fns.append(theano.function(
                inputs=input_vars + start_vars,
                outputs=[loss] + grads + next,
                on_unused_input='ignore',
                givens=givens
            ))
        else:
            grads, next = subgraph_grad(
                start=start,
                end=end,
                wrt=param_blocks[i]
            )
            # Create the grad function
            grad_fns.append(theano.function(
                inputs=input_vars + start_vars,
                outputs=grads + next,
                on_unused_input='ignore',
                givens=givens
            ))

    return grad_fns[::-1]


def compute_grads(grad_fns, param_blocks, *args):
    """
    Computes the gradients block wise.
    """
    loss = 0
    acc_grads = []
    prev = []
    for i in range(len(grad_fns) - 1, -1, -1):
        print("Evaluate block %d/%d" % (i + 1, len(grad_fns)))
        result = grad_fns[i](*args, *prev)

        if i == len(grad_fns) - 1:
            loss = result[0]
            result = result[1:]

        current = result[:len(param_blocks[i])]
        prev = result[len(param_blocks[i]):]
        acc_grads.extend(current[::-1])
    return loss, acc_grads[::-1]