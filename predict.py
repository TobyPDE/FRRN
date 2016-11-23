import lasagne
import cv2
import dltools
import theano
import theano.tensor as T
import sys
import numpy as np

sys.setrecursionlimit(10000)

config = {
    "num_classes": 19,
    "sample_factor": 2,
    "model_filename": "models/frrn_b.npz",
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
# COMPILE THEANO VAL FUNCTIONS
########################################################################################################################

with dltools.utility.VerboseTimer("Compile validation function"):
    test_predictions = lasagne.layers.get_output(network.output_layers, deterministic=True)[0]
    val_fn = theano.function(
        inputs=[input_var],
        outputs=test_predictions
    )

########################################################################################################################
# Visualize the data
########################################################################################################################

validation_provider = dltools.data.CityscapesHDDDataProvider(
    config["cityscapes_folder"],
    file_folder="val",
    batch_size=1,
    augmentor=None,
    sampling_factor=config["sample_factor"],
    random=False
)

while True:
    validation_provider.next()
    x, t = validation_provider.current()
    # Process the image
    network_output = val_fn(x)
    # Obtain a prediction
    predicted_labels = np.argmax(network_output[0], axis=0)

    prediction_visualization = dltools.utility.create_color_label_image(predicted_labels)
    ground_truth_visualization = dltools.utility.create_color_label_image(t[0])
    image = dltools.utility.tensor2opencv(x[0])

    cv2.imshow("Image", image.astype('uint8'))
    cv2.imshow("Ground Truth", ground_truth_visualization)
    cv2.imshow("Prediction", prediction_visualization)
    cv2.waitKey()
