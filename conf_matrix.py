import sys
from custom_log import FileLogReader
import numpy as np
import cityscapesutil
from termcolor import colored, cprint

cityscapesutil.init_cityscapes("/work/pohlen")
cityscapesutil.init_cityscapes("/media/toby/d/cityscapes")

log_filename = sys.argv[1]
num_labels = 19

def print_num_value(f):
    colors = ["on_blue", "on_cyan", "on_green", "on_yellow", "on_red"]
    for i in range(len(colors)):
        if f <= (i+1) * 1 / len(colors):
            print(colored(" %1.2f " % f, "white", colors[i]), end="")
            break
            
def get_training_class_labels():
    """
    Returns a list of all training class labels.
    :return: A list of training class labels.
    """
    # Import the label tool from the official toolbox
    import labels as cs_labels

    # Create map id -> color
    res = ["don't care"]
    for label in cs_labels.labels:
        if label.trainId != 255 and label.trainId != -1:
            res.append(label.name)

    return res

with FileLogReader(log_filename) as reader:
    reader.update()

    # Determine which confusion matrix to display
    command = sys.argv[2]
    if command == "best":
        # Search for the best confusion matrix in terms of IoU score
        ious = [np.average([m[i, i] / (np.sum(m[:, i]) + np.sum(m[i, :]) - m[i, i]) for i in range(0, num_labels)]) for m in reader.logs["conf_matrix"]]
        index = np.argmax(ious)
    else:
        # We expect the command to be a checkpoint index
        index = int(command)

    # Get the corresponding confusion matrix
    conf_matrix = reader.logs["conf_matrix"][index]
    
    # Compute the IoU score per class
    class_iou = [conf_matrix[i, i] / (np.sum(conf_matrix[:, i]) + np.sum(conf_matrix[i, :]) - conf_matrix[i, i]) for i in range(0, num_labels)]
    
    # Compute the pixel accuracy
    accuracy = np.diag(conf_matrix).sum() / conf_matrix.sum()

    # Get the class labels
    target_names = get_training_class_labels()[1:]

    # Normalize the matrix
    cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    #cm_normalized = cm_normalized[:-1, :-1]
    # Compute the mean class accuracy
    mean_class_accuracy = np.mean(np.diag(cm_normalized))

    # Print the matrix
    for i in range(num_labels):
        print("%13s | " % (target_names[i],), end="")
        print_num_value(class_iou[i])
        print(" | ", end="")
        
        for j in range(num_labels):
            print_num_value(cm_normalized[i, j])
        
        print("")
        """
        total_length = 13 + 2 + 4 + 4 + 19* (4 + 0)
        
        for k in range(total_length):
            print("-", end="")
        print("")
        """ 
            
    print("IoU score: %1.5f" % np.average(class_iou))
    print("Accuracy: %1.5f" % accuracy)
    print("Mean class accuracy: %1.5f" % mean_class_accuracy)
