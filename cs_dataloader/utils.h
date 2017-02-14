#include "Python.h"

#include <exception>
#include <array>
#include <vector>

/**
 * This exception is thrown when a conversion error happens.
 */
class ConversionException : public std::exception {
public:
    ConversionException(const char* message) : message(message) {}

    const char* what() const throw()
    {
        return message;
    }

private:
    /**
     * A pointer to an error message.
     */
    const char* message;
};

class PythonUtils {
public:
    /**
     * Converts a python list to a C++ vector
     */
    static void listToVector(PyObject* list, std::vector<PyObject*> & result) 
    {
        // Check if the given argument is indeed a list
        if (!PyObject_TypeCheck(list, &PyList_Type))
        {
            throw ConversionException("Provided argument is not a python list.");
        }

        const int length = PyList_GET_SIZE(list);
        result.resize(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = PyList_GET_ITEM(list, i);
        }
    }
    
    /**
     * Converts a python tuple of fixed length to a C++ array.
     */
    template<int N>
    static void tupleToArray(PyObject* tuple, std::array<PyObject*, N> & result)
    {
        // Check if the element is a tuple
        if (!PyObject_TypeCheck(tuple, &PyTuple_Type))
        {
            throw ConversionException("Provided argument is not a python tuple.");
        }

        // Check if the tuple has the correct size
        if (PyTuple_GET_SIZE(tuple) != N)
        {
            throw ConversionException("Provided tuple is not of size N.");
        }

        // Extract the two entries from the tuple
        for (int i = 0; i < N; i++) 
        {
            result[i] = PyTuple_GET_ITEM(tuple, i);
        }
    }

    /**
     * Converts a python object to string.
     */
    static std::string objectToString(PyObject* object)
    {
        // Check the type
        if (!PyObject_TypeCheck(object, &PyUnicode_Type))
        {
            throw ConversionException("Provided argument is not a python string.");
        }

        return std::string(PyUnicode_AsUTF8AndSize(object, 0));
    }

    /**
     * Converts a python object to double
     */
    static double objectToDouble(PyObject* object)
    {
        // Check the type
        if (!PyObject_TypeCheck(object, &PyFloat_Type))
        {
            throw ConversionException("Provided argument is not a python string.");
        }

        return PyFloat_AsDouble(object);
    }

    /**
     * Converts a list of string tuples to a vector of string tuples.
     */
    static void listOfStringTuplesToVector(PyObject* list, std::vector<std::pair<std::string, std::string> > & result)
    {
        std::vector<PyObject*> images;
        PythonUtils::listToVector(list, images);

        for (size_t i = 0; i < images.size(); i++)
        {
            // Get the i-th element from the list
            std::array<PyObject*, 2> entry;
            PythonUtils::tupleToArray<2>(images[i], entry);

            // It did work
            // Add the images to the list
            result.push_back(std::make_pair(
                    PythonUtils::objectToString(entry[0]),
                    PythonUtils::objectToString(entry[1])
            ));
        }
    }

    /**
     * Converts a list of doubles to a vector of doubles.
     */
    static void listOfDoublesToVector(PyObject* list, std::vector<double> & result)
    {
        std::vector<PyObject*> images;
        PythonUtils::listToVector(list, images);

        for (size_t i = 0; i < images.size(); i++)
        {
            result.push_back(PythonUtils::objectToDouble(images[i]));
        }
    }
};