#include "Python.h"
#include "numpy/arrayobject.h"

#include "lib/provider.h"
#include "utils.h"
#include "opencv2/opencv.hpp"

#include <vector>
#include <utility>
#include <string>
#include <memory>


/**
 * This is the main DataProvider class
 */
typedef struct {
    PyObject_HEAD
    std::shared_ptr<CSDataloader::DataProvider> dataProvider;
} PyDataProvider;


static std::shared_ptr<CSDataloader::IIterator<std::pair<std::string, std::string>>> makeIterator(const std::string & name, std::vector<std::pair<std::string, std::string> > & images, std::vector<double> & weights)
{
    if (name == "random")
    {
        return std::make_shared<CSDataloader::RandomIterator<std::pair<std::string, std::string>>>(images);
    }
    else if (name == "weighted")
    {
        // Make sure that there are enough weights for all examples
        if (images.size() != weights.size())
        {
            throw ConversionException("The number of weights does not match the number of images.");
        }
        return std::make_shared<CSDataloader::SampleIterator<std::pair<std::string, std::string>>>(images, weights);
    }
    else if (name == "sequential")
    {
        return std::make_shared<CSDataloader::SequentialIterator<std::pair<std::string, std::string>>>(images);
    }
    else
    {
        throw ConversionException("Unknown iterator type.");
    }
};

static std::shared_ptr<CSDataloader::IAugmentor> makeAugmentationPipeline(int subsample, double gamma, int translate)
{
    auto augmentor = std::make_shared<CSDataloader::CombinedAugmentor>();

    // Subsample the image?
    if (subsample != 1)
    {
        augmentor->addAugmentor(std::make_shared<CSDataloader::SubsampleAugmentor>(subsample));
    }

    // Cast the image to float
    augmentor->addAugmentor(std::make_shared<CSDataloader::CastToFloatAugmentor>());

    // Transform the CityScapes labels
    augmentor->addAugmentor(std::make_shared<CSDataloader::CSLabelTransformationAugmentation>());

    // Gamma augmentation
    if (gamma != 0.0)
    {
        augmentor->addAugmentor(std::make_shared<CSDataloader::GammaAugmentor>(gamma));
    }

    // Translation augmentation
    if (translate != 0)
    {
        augmentor->addAugmentor(std::make_shared<CSDataloader::TranslationAugmentor>(translate));
    }

    return augmentor;
}

/**
        " - imgs         A list of string tuples of filenames. The first entry of each\n"
        "                tuple is considered to be the image filename while\n"
        "                the second entry is considered to be the target image filename.\n"
        " - iterator     A string indicating which iterator to use. Possible values:\n"
        "                'sequential', 'random', 'weighted'\n"
        "                - 'sequential' means that we iterate over the dataset in a\n"
        "                  sequential deterministic order\n"
        "                - 'random' means that we iterate over the dataset in a random\n"
        "                  order\n"
        "                - 'weighted' means that we don't iterate over the dataset but\n"
        "                   simply draw examples according to the given distribution.\n"
        " - weighted     A list of datapoint weights.\n"
        " - translate    An integer that determines the strength or the translation\n"
        "                augmentation. If this is not provided, no\n"
        "                translation augmentation is applied.\n"
        " - gamma        A floating point value in [0, 0.5] that determines the\n"
        "                strength or the gamma augmentation. If no parameter is\n"
        "                provided, no gamma augmentation is performed.\n"
        " - subsample    The subsampling factor. Must be a multiple of 2. If no\n"
        "                subsampling factor is provided, the images are not\n"
        "                subsampled.\n"
        " - batchsize    The batchsize to use. The default is 1.\n"
 * @param self
 * @param args
 * @param keywords
 * @return
 */
static int PyDataProvider_init(PyDataProvider* self, PyObject* args, PyObject *keywords)
{
    static char *keywordList[] = {"imgs", "iterator", "weights", "translate", "gamma", "subsample", "batchsize", 0};

    PyObject* imgs;
    char * iterator = "random";
    PyObject* weights = 0;
    int translate = 0;
    double gamma = 0.0;
    int subsample = 1;
    int batchSize = 1;

    if (!PyArg_ParseTupleAndKeywords(args, keywords, "O!|sO!idii", keywordList, &PyList_Type, &imgs, &iterator,
                                     &PyList_Type, &weights, &translate, &gamma, &subsample, &batchSize))
    {
        return 0;
    }

    // Extract the list of images
    try {
        std::vector<std::pair<std::string, std::string> > extractedImages;
        std::vector<double> extractedWeights;

        PythonUtils::listOfStringTuplesToVector(imgs, extractedImages);
        if (weights != nullptr)
        {
            PythonUtils::listOfDoublesToVector(weights, extractedWeights);
        }

        // Create the iterator to use
        auto dataIterator = makeIterator(iterator, extractedImages, extractedWeights);

        // Create the augmentation pipeline
        auto augmentor = makeAugmentationPipeline(subsample, gamma, translate);

        // Create the data provider
        auto dataProvider = std::make_shared<CSDataloader::DataProvider>(dataIterator, batchSize, augmentor);

        // Create the data provider
        self->dataProvider = dataProvider;

        return 0;
    }
    catch (ConversionException & e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    }
}

static PyObject* PyDataProvider_next(PyDataProvider* self)
{
    if (self->dataProvider != nullptr && self->dataProvider->getBatchSize() > 0)
    {
        // Get the next batch
        auto batch = self->dataProvider->next();

        // Convert the batch to numpy arrays
        npy_intp imgDims[4] = {0, 3, 0, 0};
        imgDims[0] = static_cast<npy_intp>(self->dataProvider->getBatchSize());
        imgDims[2] = batch->at(0).img.rows;
        imgDims[3] = batch->at(0).img.cols;
        PyArrayObject* imgs = (PyArrayObject*) PyArray_ZEROS(4, imgDims, NPY_FLOAT, 0);

        npy_intp targetDims[3] = {0, 0, 0};
        targetDims[0] = static_cast<npy_intp>(self->dataProvider->getBatchSize());
        targetDims[1] = batch->at(0).img.rows;
        targetDims[2] = batch->at(0).img.cols;
        PyArrayObject* target = (PyArrayObject*) PyArray_ZEROS(3, targetDims, NPY_INT, 0);

        for (int i0 = 0; i0 < imgDims[0]; i0++) {
            for (int i2 = 0; i2 < imgDims[2]; i2++) {
                for (int i3 = 0; i3 < imgDims[3]; i3++) {
                    npy_int* t = (npy_int*) PyArray_GETPTR3(target, i0, i2, i3);
                    // Convert 255 to -1 (void labels)
                    const uchar label = batch->at(i0).target.at<uchar>(i2, i3);;
                    *t = label == 255 ? -1 : label;

                    const cv::Vec3f & img = batch->at(i0).img.at<cv::Vec3f>(i2, i3);
                    for (int i1 = 0; i1 < imgDims[1]; i1++) {
                        // Convert it to RGB
                        npy_float * x = (npy_float *) PyArray_GETPTR4(imgs, i0, 2 - i1, i2, i3);
                        *x = img[i1];
                    }
                }
            }
        }

        // Create the return type (tuple)
        PyObject* result = PyTuple_New(2);
        PyTuple_SetItem(result, 0, (PyObject*) imgs);
        PyTuple_SetItem(result, 1, (PyObject*) target);

        return result;
    }
    else
    {
        Py_RETURN_NONE;
    }
}

static PyObject* PyDataProvider_reset(PyDataProvider* self)
{
    if (self->dataProvider != nullptr)
    {
        self->dataProvider->reset();
    }
    Py_RETURN_NONE;
}

static PyObject* PyDataProvider_getNumBatches(PyDataProvider* self)
{
    if (self->dataProvider != nullptr)
    {
        PyObject* result = PyLong_FromLong(self->dataProvider->getNumBatches());
        return result;
    }
    Py_RETURN_NONE;
}

static PyMethodDef PyDataProvider_methods[] = {
        {"next", (PyCFunction)PyDataProvider_next, METH_NOARGS,
                "Returns the next batch"
        },
        {"reset", (PyCFunction)PyDataProvider_reset, METH_NOARGS,
                "Resets the iterator"
        },
        {"get_num_batches", (PyCFunction)PyDataProvider_getNumBatches, METH_NOARGS,
                "Returns the number of batches"
        },
        {NULL}  /* Sentinel */
};

static PyTypeObject PyDataProviderType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "cs_dataloader.DataProvider",             /* tp_name */
        sizeof(PyDataProvider), /* tp_basicsize */
        0,                         /* tp_itemsize */
        0,                         /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,        /* tp_flags */
        "A data provider class for the cityscapes dataset",           /* tp_doc */
        0,                         /* tp_traverse */
        0,                         /* tp_clear */
        0,                         /* tp_richcompare */
        0,                         /* tp_weaklistoffset */
        0,                         /* tp_iter */
        0,                         /* tp_iternext */
        PyDataProvider_methods,      /* tp_methods */
        0,                         /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)PyDataProvider_init,                         /* tp_init */
        0,                         /* tp_alloc */
        0,                         /* tp_new */};


static PyMethodDef CsDataLoaderMethods[] = {
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cs_dataloadermodule = {
        PyModuleDef_HEAD_INIT,
        "cs_dataloader",   /* name of module */
        0, /* module documentation, may be NULL */
        -1,
        CsDataLoaderMethods
};

PyMODINIT_FUNC PyInit_cs_dataloader(void)
{
    PyObject* m;

    PyDataProviderType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PyDataProviderType) < 0)
        return NULL;

    m = PyModule_Create(&cs_dataloadermodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyDataProviderType);
    PyModule_AddObject(m, "DataProvider", (PyObject *)&PyDataProviderType);
    import_array();

    return m;
}