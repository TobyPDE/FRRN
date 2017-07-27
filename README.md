# Full-Resolution Residual Networks (FRRN)

This repository contains code to train and qualitatively evaluate 
*Full-Resolution Residual Networks (FRRNs)* as described in

**Tobias Pohlen, Alexander Hermans, Markus Mathias, Bastian Leibe: *Full***
***Resolution Residual Networks for Semantic Segmentation in Street Scenes.***
**CVPR 2017.**

A pre-print of the paper can be found on arXiv: 
[arXiv:1611.08323](https://arxiv.org/abs/1611.08323).

Please cite the work as follows:

```bibtex
@inproceedings{pohlen2017FRRN,
  title={Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes},
  author={Pohlen, Tobias and Hermans, Alexander and Mathias, Markus and Leibe, Bastian},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

## Demo  Video
[Click here to watch our video](https://www.youtube.com/watch?v=PNzQ4PNZSzc).

## Installation

Install the following software packages:

* Python 2.7 or 3.4
* Numpy
* Scipy
* Scikit-Learn
* OpenCV
* Theano
  * Scipy
  * Scikit-Learn
* Lasagne

You may *optionally* install the following library for better performance. 

* Chianti [https://github.com/TobyPDE/chianti](https://github.com/TobyPDE/chianti)

You can check if all dependencies are installed correctly by running the 
`check_dependencies.py` script:

```bash
$ python check_dependencies.py --cs_folder=[Your CS folder]
2017-07-26 22:17:34,945 INFO Found supported Python version 3.4.
2017-07-26 22:17:35,122 INFO Successfully imported numpy.
2017-07-26 22:17:35,184 INFO Successfully imported cv2.
2017-07-26 22:17:35,666 INFO Successfully imported sklearn.
2017-07-26 22:17:35,691 INFO Successfully imported sklearn.metrics.
2017-07-26 22:17:35,691 INFO Successfully imported scipy.
Using cuDNN version 6021 on context None
Mapped name None to device cuda: TITAN X (Pascal) (0000:02:00.0)
2017-07-26 22:17:38,760 INFO Successfully imported theano.
2017-07-26 22:17:38,797 INFO Successfully imported lasagne.
2017-07-26 22:17:38,797 INFO Theano float is float32.
2017-07-26 22:17:38,803 INFO cuDNN spatial softmax found.
2017-07-26 22:17:38,807 INFO Use Chianti C++ library.
2017-07-26 22:17:38,826 INFO Found CityScapes training set.
2017-07-26 22:17:38,826 INFO Found CityScapes validation set.
```

If you don't see any `ERROR` messages, the software should run on your machine.

## Qualitatively evaluation a pre-trained model

Run the script `predict.py`.

```bash
$ python predict.py --help
usage: predict.py [-h] --architecture {frrn_a,frrn_b} --model_file MODEL_FILE
                  --cs_folder CS_FOLDER [--sample_factor SAMPLE_FACTOR]

Shows the predictions of a Full-Resolution Residual Network on the Cityscapes
validation set.

optional arguments:
  -h, --help            show this help message and exit
  --architecture {frrn_a,frrn_b}
                        The network architecture type.
  --model_file MODEL_FILE
                        The model filename. Weights are initialized to the
                        given values if the file exists. Snapshots are stored
                        using a _snapshot_[iteration] post-fix.
  --cs_folder CS_FOLDER
                        The folder that contains the Cityscapes Dataset.
  --sample_factor SAMPLE_FACTOR
                        The sampling factor.
```

## Train a new model

Run the `train.py` script.

```bash 
$ python train.py --help
usage: train.py [-h] --architecture {frrn_a,frrn_b,frrn_c} --model_file
                MODEL_FILE --log_file LOG_FILE --cs_folder CS_FOLDER
                [--batch_size BATCH_SIZE]
                [--validation_interval VALIDATION_INTERVAL]
                [--iterator {uniform,weighted}] [--crop_size CROP_SIZE]
                [--learning_rate LEARNING_RATE]
                [--sample_factor SAMPLE_FACTOR]

Trains a Full-Resolution Residual Network on the Cityscapes Dataset.

optional arguments:
  -h, --help            show this help message and exit
  --architecture {frrn_a,frrn_b}
                        The network architecture type.
  --model_file MODEL_FILE
                        The model filename. Weights are initialized to the
                        given values if the file exists. Snapshots are stored
                        using a _snapshot_[iteration] post-fix.
  --log_file LOG_FILE   The log filename. Use log_monitor.py in order to
                        monitor training progress in the terminal.
  --cs_folder CS_FOLDER
                        The folder that contains the Cityscapes Dataset.
  --batch_size BATCH_SIZE
                        The batch size.
  --validation_interval VALIDATION_INTERVAL
                        The validation interval.
  --iterator {uniform,weighted}
                        The dataset iterator type.
  --crop_size CROP_SIZE
                        The size of crops to extract from the full-resolution
                        images. If 0, then now crops will be extracted.
  --learning_rate LEARNING_RATE
                        The learning rate to use.
  --sample_factor SAMPLE_FACTOR
                        The sampling factor.
```

## Monitor training

Start a new notebook server and open `training_monitor.ipynb`.

## License

See `LICENSE` (MIT).

## Copyright

Copyright (c) 2017 Google Inc. 

Copyright (c) 2017 Toby Pohlen

