import theano.tensor as T

from . import utility


class MiniBatchOptimizer(object):
    """
    This class optimizes a neural network using Mini batches.
    """

    def __init__(self, train_fn, data_sources, hooks):
        """
        Initializes a new instance of the MiniBatchOptimizer class.

        :param train_fn: The function that updates the network parameters. It has to return a list of losses.
        :param data_sources: A list of data sources that feed into the train_fn.
        :param hooks: A list of hooks that are called at certain points in the training process.
        """
        self.train_fn = train_fn
        self.data_sources = data_sources
        self.hooks = hooks

    def call_hooks(self, **kwargs):
        """
        Calls all hooks that have been registered.

        :param kwargs: A list of parameters that is passed to all hooks.
        """
        for h in self.hooks:
            try:
                h.update(**kwargs)
            except Exception as e:
                print("Cannot run hook %s" % h)
                print("Error message: %s" % e)

    def optimize(self):
        """
        Runs the main optimization loop until it is interrupted by the user via CTRL+C.
        """
        update_counter = -1

        # Start the optimization
        with utility.Uninterrupt() as u:
            while not u.interrupted:
                update_counter += 1

                # Advance the data_source iterators
                # Gather the arguments for the training function
                train_fn_args = []
                for s in self.data_sources:
                    s.next()
                    train_fn_args.extend(s.current())

                with utility.Timer() as t:
                    losses = self.train_fn(*train_fn_args)

                self.call_hooks(
                    update_counter=update_counter,
                    losses=losses,
                    runtime=t.interval)


def get_gradient_variables(params):
    """
    Creates shared variables in order to accumulate the gradients off the GPU.
    :param params: A list of trainable parameters.
    :return: A list of gradient variables.
    """
    return [T.zeros_like(p) for p in params]

