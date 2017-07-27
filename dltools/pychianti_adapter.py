"""API adapter to replace pychianti with the data module.

This class provides an API adapter that makes the data module's API similar to
pychianti's API. This allows the user to use most parts of the code without
having to install chianti.
"""

from dltools import data


DataProvider = data.DataProvider


class Iterator(object):
    @staticmethod
    def WeightedRandom(sequence, weights):
        return data.WeightedRandomIterator(sequence, weights)

    @staticmethod
    def Random(sequence):
        return data.RandomIterator(sequence)

    @staticmethod
    def Sequential(sequence):
        return data.SequentialIterator(sequence)


class Augmentor(object):
    @staticmethod
    def Subsample(factor):
        return data.SubsampleAugmentor(factor)

    @staticmethod
    def Crop(size, num_classes):
        return data.CropAugmentor(size, num_classes)

    @staticmethod
    def Translation(offset):
        return data.TranslationAugmentor(offset)

    @staticmethod
    def Zooming(factor):
        return data.ZoomingAugmentor(factor)

    @staticmethod
    def Gamma(factor):
        return data.GammaAugmentor(factor)

    @staticmethod
    def Saturation(min_delta, max_delta):
        return data.SaturationAugmentor(min_delta, max_delta)

    @staticmethod
    def Hue(min_delta, max_delta):
        return data.HueAugmentor(min_delta, max_delta)

    @staticmethod
    def Rotation(max_angel):
        return data.RotationAugmentor(max_angel)

    @staticmethod
    def Combined(augmentors):
        return data.CombinedAugmentor(augmentors)


class Loader(object):
    @staticmethod
    def RGB():
        return data.RGBLoader()

    @staticmethod
    def ValueMapper(value_map):
        return data.ValueMapperLoader(value_map)


