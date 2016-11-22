import cv2
import numpy as np
import lasagne
import theano
import theano.tensor as T
import sys
import dltools

sys.setrecursionlimit(10000)

sample_factor = 2
model_filename = sys.argv[1]
use_runtime_augmentation = True

########################################################################################################################
# LOAD THE DATA
########################################################################################################################

with dltools.utility.VerboseTimer("Load data"):
    val_images = dltools.utility.load_np_data_array("data/city_scapes_val_images_down_%d.npz" % sample_factor)
    val_labels = dltools.utility.load_np_data_array("data/city_scapes_val_labels_down_%d.npz" % sample_factor)

########################################################################################################################
# DEFINE THE NETWORK
########################################################################################################################

with dltools.utility.VerboseTimer("Define network"):
    # Define the theano variables
    input_var = T.ftensor4()

    builder = dltools.architectures.FRRNBBuilder(
        base_channels=48,
        lanes=32,
        multiplier=2,
        num_classes=19,
        use_bilinear_upscaling=True
    )
    network = builder.build(
        input_var=input_var,
        input_shape=(None, 3, 512, 1024))


#######################################################################################################################
# LOAD MODEL
########################################################################################################################

with dltools.utility.VerboseTimer("Load model"):
    network.load_model(model_filename)

########################################################################################################################
# COMPILE THEANO VAL FUNCTIONS
########################################################################################################################

with dltools.utility.VerboseTimer("Compile validation function"):
    test_predictions = lasagne.layers.get_output(network.output_layers, deterministic=True)[0]
    eval_fn = theano.function(
        inputs=[input_var],
        outputs=test_predictions
    )

indices = np.arange(len(val_images))
#np.random.shuffle(indices)

for j in range(len(val_images)):
    i = indices[j]
    # Process the image
    network_output = np.array(eval_fn(val_images[i:i + 1])[0])
    # Obtain a prediction
    predicted_labels = np.argmax(network_output, axis=0)

    prediction_visualization = dltools.utility.create_color_label_image(predicted_labels)
    ground_truth_visualization = dltools.utility.create_color_label_image(val_labels[i])
    image = dltools.utility.tensor2opencv(val_images[i])

    cv2.imshow("Image", image)
    cv2.imshow("Ground Truth", ground_truth_visualization)
    cv2.imshow("Prediction", prediction_visualization)
    cv2.waitKey()
