from distutils.core import setup, Extension
import numpy as np

module = Extension('cs_dataloader',
                   libraries=['opencv_core', 'opencv_highgui', 'opencv_imgproc', 'opencv_imgcodecs'],
                   include_dirs = [np.get_include()],
                   extra_link_args=['-lgomp'],
                   extra_compile_args=['-std=c++11', '-fopenmp'],
                   sources=['dataprovider.cpp'])

setup( name='Cityscapes Dataloader',
       version='1.0',
       description='This package loads batches of images from the CityScapes dataset asynchonously.',
       author='Tobias Pohlen',
       author_email='tobias.pohlen@rwth-aachen.de',
       ext_modules=[module])