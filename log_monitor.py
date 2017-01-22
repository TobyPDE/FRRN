import sys
from dltools.logging import FileLogReader
import numpy as np
import time

history = 25
log_filename = sys.argv[1]

reader = FileLogReader(log_filename)

while True:
    reader.update()
    format = "%30s: %3.6f"

    print("\n")
    print("=" * 40)
    try:
        for s in reader.logs["status"][-5:]:
            print("%s" % s)
    except:
        pass

    print("")
    print("Training status")
    print("===============")

    # Print the number of processed batches
    try:
        print(format % ("Num processed batches", reader.logs["update_counter"][-1]))
    except:
        pass

    # Print the average training error for the last 50 iterations
    try:
        avg_losses = np.mean(reader.logs["losses"][-history:], axis=0)
        print(format % ("Training error", avg_losses))
    except Exception as e:
        print(e)
        pass

    # Print the average update times for the last 50 iterations
    try:
        print(format % ("Data times", np.average(reader.logs["data_runtime"][-history:])))
    except:
        pass
    try:
        print(format % ("Update times", np.average(reader.logs["update_runtime"][-history:])))
    except:
        pass

    print("")
    print("Last validation")
    print("===============")

    # Print the average validation loss for the last 50 iterations
    try:
        print(format % ("Validation loss", float(reader.logs["validation_loss"][-1])))
    except:
        pass

    try:
        # Get all confusion matrices
        matrices = [np.asarray(m) for m in reader.logs["conf_matrix"]]
        num_labels = matrices[0].shape[0]
        ious = [np.average([m[i, i] / (np.sum(m[:, i]) + np.sum(m[i, :]) - m[i, i]) for i in range(0, num_labels)]) for m in
                matrices]
        pixel_acc_scores = [np.sum(np.diag(m)) / np.sum(m) for m in reader.logs["conf_matrix"]]

        # Print the last validation accuracy
        print(format % ("Validation accuracy", pixel_acc_scores[-1]))

        # Print the last IoU score
        print(format % ("IoU score", ious[-1]))
    except:
        pass

    print("")
    print("Best checkpoint (acc)")
    print("=====================")

    # Print the best validation accuracy
    # Determine the index of the best pixel accuracy
    try:
        best_pixel_acc_index = int(np.argmax(pixel_acc_scores))
        print(format % ("Best validation accuracy", pixel_acc_scores[best_pixel_acc_index]))
        print(format % ("Corresponding IoU score", ious[best_pixel_acc_index]))
        print(format % ("Index", float(reader.logs["validation_checkpoint"][best_pixel_acc_index])))
    except:
        pass

    print("")
    print("Best checkpoint (IoU)")
    print("=====================")

    # Print the best IoU score
    # Determine the index of the best IoU score
    try:
        best_iou_index = int(np.argmax(ious))
        #for i in range(len(ious)):
        #    print(ious[i])
        print(format % ("Best IoU score", ious[best_iou_index]))
        print(format % ("Corresponding accuracy", pixel_acc_scores[best_iou_index]))
        print(format % ("Index", float(reader.logs["validation_checkpoint"][best_iou_index])))
    except:
        pass

    time.sleep(1)