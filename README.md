# Full Resolution Residual Networks (FRRN)

This repository contains code to train and qualitatively evaluate 
*Full Resolution Residual Networks (FRRNs)* as described in
**Tobias Pohlen, Alexander Hermans, Markus Mathias, Bastian Leibe. Full Resolution Residual Networks for Semantic Segmentation in Street Scenes. [arXiv:1611.08323](https://arxiv.org/abs/1611.08323)**. 

## Demo  Video
[Click here to watch our video](https://www.youtube.com/watch?v=PNzQ4PNZSzc).

## Prerequisite

In order to run the code, your setup has to meet the following minimum requirements (tested versions in parentheses. Other versions might work, too):

* Python 3.5 (3.5)
    * Chianti [https://github.com/TobyPDE/chianti](https://github.com/TobyPDE/chianti)
    * Theano (0.9.0.dev1) with cuDNN (5004)
    * Lasagne (0.2.dev1)
    * OpenCV (3.1.0)
    * Scikit-Learn (0.17.1)
    * Numpy (1.11.0)

## How do I evaluate the pre-trained model?

Step by step guide:

1. Download and extract the files *gtFine_trainvaltest.zip* and *leftImg8bit_trainvaltest.zip (11GB)* from the [CityScapes website](https://www.cityscapes-dataset.com/downloads/). 
2. Clone the repository
3. Change into the *FRRN* directory and run `$ python predict_frrn_[a|b].py`
4. When being asked, enter the path to the directory where you extracted the dataset to and choose the default model file. 

## How do I train a new model?

Step by step guide:

1. Download and extract the files *gtFine_trainvaltest.zip* and *leftImg8bit_trainvaltest.zip (11GB)* from the [CityScapes website](https://www.cityscapes-dataset.com/downloads/). 
2. Clone the repository
3. Inspect the configuration at the beginning of the file `train_frrn_[a|b].py` and change it if necessary.
4. Run `$ python train_frrn_[a|b].py`
5. When being asked, enter the path to the directory where you extracted the dataset to and choose a new model file and new log file. 

## How do I monitor training?

Simply run `$ python log_monitor.py [your log file]`. 

## License

The MIT License (MIT)
Copyright (c) 2016 Tobias Pohlen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
