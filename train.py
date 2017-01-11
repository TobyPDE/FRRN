import lasagne

import dltools
import theano
import theano.tensor as T
import sys
import numpy as np

sys.setrecursionlimit(10000)

config = {
    "num_classes": 19,
    "batch_size": 3,
    "sample_factor": 2,
    "validation_frequency": 500,
    "validation_batch_size": 3,
    # Recall that during validation, the entire batch is passed through
    # the whole network. If your GPU memory does not suffice, this will
    # crash the process.
    "model_filename": "models/frrn_b.npz",
    "log_filename": "logs/frrn_b.log",
    "snapshot_frequency": 500,
    "base_channels": 48,
    "fr_channels": 32,
    "cityscapes_folder": "/"
}

########################################################################################################################
# Ask for the cityscapes path
########################################################################################################################

config["cityscapes_folder"] = dltools.utility.get_interactive_input(
    "Enter path to CityScapes folder",
    "cityscapes_folder.txt",
    config["cityscapes_folder"])

config["model_filename"] = dltools.utility.get_interactive_input(
    "Enter model filename",
    "model_filename.txt",
    config["model_filename"])

config["log_filename"] = dltools.utility.get_interactive_input(
    "Enter log filename",
    "log_filename.txt",
    config["log_filename"])

########################################################################################################################
# DEFINE THE NETWORK
########################################################################################################################

with dltools.utility.VerboseTimer("Define network"):
    # Define the theano variables
    input_var = T.ftensor4()

    builder = dltools.architectures.FRRNBBuilder(
        base_channels=config["base_channels"],
        lanes=config["fr_channels"],
        multiplier=2,
        num_classes=config["num_classes"]
    )
    network = builder.build(
        input_var=input_var,
        input_shape=(None, 3, 1024 // config["sample_factor"], 2048 // config["sample_factor"]))

#######################################################################################################################
# LOAD MODEL
########################################################################################################################

with dltools.utility.VerboseTimer("Load model"):
    network.load_model(config["model_filename"])
    
########################################################################################################################
# DEFINE LOSS
########################################################################################################################

with dltools.utility.VerboseTimer("Define loss"):
    # Get the raw network outputs
    target_var = T.itensor3()

    # Get the original predictions back
    # Set deterministic=False if you want to train with batch norm enabled
    all_predictions, split_outputs = dltools.hybrid_training.get_split_outputs(network, deterministic=False)
    predictions = all_predictions[0]

    test_all_outputs = lasagne.layers.get_output(network.output_layers, deterministic=True)
    test_predictions = test_all_outputs[0]

    # Training classification loss (supervised)
    classification_loss = dltools.utility.bootstrapped_categorical_cross_entropy4d_loss(
        predictions,
        target_var,
        batch_size=config["batch_size"],
        multiplier=64)

    # Validation classification loss (supervised)
    test_classification_loss = dltools.utility.bootstrapped_categorical_cross_entropy4d_loss(
        test_predictions,
        target_var,
        batch_size = config["validation_batch_size"],
        multiplier = 64)

    loss = classification_loss

########################################################################################################################
# COMPILE THEANO TRAIN FUNCTIONS
########################################################################################################################

with dltools.utility.VerboseTimer("Compile update functions"):
    param_blocks, params = dltools.hybrid_training.split_params(network)
    grad_fns = dltools.hybrid_training.compile_grad_functions(
        split_outputs,
        param_blocks,
        [input_var, target_var],
        loss,
        {})

    # Optimization parameters
    learning_rate = T.fscalar()

    # Create the update function
    grad_vars = dltools.optimizer.get_gradient_variables(params)

    # Choose whatever optimizer you like
    updates = lasagne.updates.adam(grad_vars, params, learning_rate=learning_rate)

    update_fn = theano.function(
        inputs=[learning_rate] + grad_vars,
        updates=updates,
    )

    update_counter = 1

    def compute_update(*args):
        global update_counter

        update_counter += 1

        # Compute the learning rate
        lr = np.float32(1e-3)
        if update_counter > 60000:
            lr = np.float32(1e-4)
        
        # Compute all gradients
        loss, grads = dltools.hybrid_training.compute_grads(grad_fns, param_blocks, *args)
        print("loss=%e, learning_rate=%e, update_counter=%d" % (loss, lr, update_counter))
        update_fn(lr, *grads)
        return loss

########################################################################################################################
# COMPILE THEANO VAL FUNCTIONS
########################################################################################################################

with dltools.utility.VerboseTimer("Compile validation function"):
    val_fn = theano.function(
        inputs=[input_var, target_var],
        outputs=[T.argmax(test_predictions, axis=1), test_classification_loss]
    )

########################################################################################################################
# SET UP OPTIMIZER
########################################################################################################################

with dltools.utility.VerboseTimer("Optimize"):
    logger = dltools.logging.FileLogWriter(config["log_filename"])

    provider = dltools.data.CityscapesHDDDataProvider(
        config["cityscapes_folder"],
        file_folder="train",
        batch_size=config["batch_size"],
        augmentor=dltools.augmentation.CombinedAugmentor([
            dltools.augmentation.CastAugmentor(),
            dltools.augmentation.TranslationAugmentor(offset=40),
            dltools.augmentation.GammaAugmentor(gamma_range=(-0.05, 0.05))
        ]), 
        sampling_factor=config["sample_factor"]
    )

    validation_provider = dltools.data.CityscapesHDDDataProvider(
        config["cityscapes_folder"],
        file_folder="val",
        batch_size=config["validation_batch_size"],
        augmentor=None,
        sampling_factor=config["sample_factor"],
        random=False
    )

    optimizer = dltools.optimizer.MiniBatchOptimizer(
        compute_update,
        [
            provider
        ],
        [
            dltools.hooks.SnapshotHook(config["model_filename"], network, frequency=config["snapshot_frequency"]),
            dltools.hooks.LoggingHook(logger),
            dltools.hooks.SegmentationValidationHook(
                val_fn,
                validation_provider,
                logger,
                frequency=config["validation_frequency"])
        ])
    optimizer.optimize()
