import lasagne

import dltools
import theano
import theano.tensor as T
import sys
import numpy as np

sys.setrecursionlimit(10000)

config = {
    "num_classes": 19,
    "batch_size": 1,
    "sample_factor": 4,
    "validation_frequency": 500,
    "validation_batch_size": 3,
    "model_filename": "models/frrn_c.npz",
    "log_filename": "logs/frrn_c.log",
    "snapshot_frequency": 500,
    "base_channels": 48,
    "fr_channels": 32,
    "data": "data/cityscapes_4x.npz"
}

########################################################################################################################
# Ask for the cityscapes path
########################################################################################################################

config["data"] = dltools.utility.get_interactive_input(
    "Enter path to data file",
    "data.txt",
    config["data"])

config["model_filename"] = dltools.utility.get_interactive_input(
    "Enter model filename",
    "model_filename_2.txt",
    config["model_filename"])

config["log_filename"] = dltools.utility.get_interactive_input(
    "Enter log filename",
    "log_filename_2.txt",
    config["log_filename"])

########################################################################################################################
# Load the dataset
########################################################################################################################

with dltools.utility.VerboseTimer("Load data"):
    with np.load(config["data"]) as data:
        imgs_train, targets_train, imgs_val, targets_val = data['arr_0']

########################################################################################################################
# DEFINE THE NETWORK
########################################################################################################################

with dltools.utility.VerboseTimer("Define network"):
    # Define the theano variables
    input_var = T.ftensor4()

    builder = dltools.architectures.FRRNCBuilder(
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
    all_predictions = lasagne.layers.get_output(network.output_layers, deterministic=False)
    predictions = all_predictions[0]

    test_all_outputs = lasagne.layers.get_output(network.output_layers, deterministic=True)
    test_predictions = test_all_outputs[0]

    # Training classification loss (supervised)
    classification_loss = dltools.utility.bootstrapped_categorical_cross_entropy4d_loss(
        predictions,
        target_var,
        batch_size=config["batch_size"],
        multiplier=32)

    # Validation classification loss (supervised)
    test_classification_loss = dltools.utility.bootstrapped_categorical_cross_entropy4d_loss(
        test_predictions,
        target_var,
        batch_size=config["validation_batch_size"],
        multiplier=32)

    loss = classification_loss

########################################################################################################################
# COMPILE THEANO TRAIN FUNCTIONS
########################################################################################################################

with dltools.utility.VerboseTimer("Compile update functions"):
    learning_rate = T.fscalar()

    # Choose whatever optimizer you like
    params = lasagne.layers.get_all_params(network.output_layers, trainable=True)
    updates = lasagne.updates.adam(classification_loss, params, learning_rate=learning_rate)

    update_fn = theano.function(
        inputs=[learning_rate, input_var, target_var],
        outputs=classification_loss,
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
        loss = update_fn(lr, *args)
        print("loss=%e, learning_rate=%e, update_counter=%d" % (loss, lr, update_counter))
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

    provider = dltools.data.DataProvider(
        [imgs_train, targets_train],
        batch_size=config["batch_size"],
        augmentor=dltools.augmentation.CombinedAugmentor([
            dltools.augmentation.CastAugmentor(),
            dltools.augmentation.TranslationAugmentor(offset=20),
            dltools.augmentation.GammaAugmentor(gamma_range=(-0.05, 0.05))
        ])
    )

    validation_provider = dltools.data.DataProvider(
        [imgs_val, targets_val],
        batch_size=config["validation_batch_size"],
        augmentor=None,
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
