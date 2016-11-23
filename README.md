# Full Resolution Residual Networks (FRRN)

This repository contains code to train and qualitatively evaluate 
*Full Resolution Residual Networks (FRRNs)* as described in
**Tobias Pohlen, Alexander Hermans, Markus Mathias, Bastian Leibe. Full Resolution Residual Networks for Semantic Segmentation in Street Scenes. [arXiv:xxxxxxx](http://arxiv.org/abs/xxxxxx)**. 

## Prerequisite

In order to run the code, your setup has to meet the following minimum requirements:

* Python 3
    * Theano with CUDA
    * Lasagne
    * OpenCV
    * Scikit-Learn
    * Termcolor
    * Numpy

## How do I evaluate the pre-trained model?

Step by step guide:

1. Download and extract the files *gtFine_trainvaltest.zip* and *leftImg8bit_trainvaltest.zip (11GB)* from the [CityScapes website](https://www.cityscapes-dataset.com/downloads/). 
2. Clone the repository
3. Change into the *FRRN* directory and run `$ python predict.py`
4. When being asked, enter the path to the directory where you extracted the dataset to and choose the default model file. 

## How do I train a new model?

Step by step guide:

1. Download and extract the files *gtFine_trainvaltest.zip* and *leftImg8bit_trainvaltest.zip (11GB)* from the [CityScapes website](https://www.cityscapes-dataset.com/downloads/). 
2. Clone the repository
3. Inspect the configuration at the beginning of the file `train.py` and change it if necessary.
3. Run `$ python train.py`


## License

The MIT License (MIT)
Copyright (c) 2016 Tobias Pohlen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.