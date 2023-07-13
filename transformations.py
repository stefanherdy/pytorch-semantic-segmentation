#!/usr/bin/python

import numpy as np
from sklearn.externals._pilutil import bytescale
import random
import matplotlib.pyplot as plt
import cv2
import random



def create_dense_target(tar: np.ndarray):
    classes = np.unique(tar)
    # print(unique)
    dummy = tar[:,:]
    return dummy


def normalize_01(inp: np.ndarray):
    mi = np.min(inp)
    ma = np.max(inp)
    range = np.ptp(inp)
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    return inp_out


def normalize(inp: np.ndarray, mean: float, std: float):
    inp_out = (inp - mean) / std
    return inp_out


def re_normalize(inp: np.ndarray,
                 low: int = 0,
                 high: int = 255
                 ):
    """Normalize the data to a certain range. Default: [0-255]"""
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out


class Compose:
    """
    Composes several transforms together.
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, inp, target):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target

    def __repr__(self): return str([transform for transform in self.transforms])


class MoveAxis:
    """From [H, W, C] to [C, H, W]"""

    def __init__(self, transform_input: bool = True, transform_target: bool = False):
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        inp = np.moveaxis(inp, -1, 0)
        #tar = np.moveaxis(tar, -1, 0)
        

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class DenseTarget:
    """Creates segmentation maps with consecutive integers, starting from 0"""

    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        tar = create_dense_target(tar)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class RandomFlip:

    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        rand = random.choice([0, 1])
        if rand == 1:
            #inp = np.ndarray.copy(np.fliplr(inp))
            inp = np.moveaxis(inp, 0, -1)
            inp = cv2.flip(inp, 1)
            inp = np.moveaxis(inp, -1, 0)
            tar = np.ndarray.copy(np.fliplr(tar))

        rand = random.choice([0, 1])
        if rand == 1:
            #inp = np.ndarray.copy(np.flipud(inp, axis=(1,2)))
            inp = np.moveaxis(inp, 0, -1)
            inp = cv2.flip(inp, 0)
            inp = np.moveaxis(inp, -1, 0)
            tar = np.ndarray.copy(np.flipud(tar))

        rand = random.choice([0, 1])
        if rand == 1:
            inp = np.ndarray.copy(np.rot90(inp, k=1, axes=(1, 2)))
            tar = np.ndarray.copy(np.rot90(tar, k=1, axes=(0, 1)))
        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class RandomCrop:

    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):

        crop_width = 512
        crop_height =512

        max_x = inp.shape[1] - crop_width
        print(max_x)
        max_y = inp.shape[2] - crop_height
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        inp = np.moveaxis(inp, 0, -1)
        inp = inp[x: x + crop_width, y: y + crop_height,:]
        inp = np.moveaxis(inp, -1, 0)
        tar = tar[x: x + crop_width, y: y + crop_height]

        return inp, tar


class Resize_Sample:

    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):

        
        inp = np.moveaxis(inp, 0, -1)
        inp = cv2.resize(inp, (256,256), interpolation = cv2.INTER_NEAREST)
        inp = np.moveaxis(inp, -1, 0)
        tar = cv2.resize(tar, (256,256), interpolation = cv2.INTER_NEAREST)

        return inp, tar


class Normalize01:
    """Squash image input to the value range [0, 1] (no clipping)"""

    def __init__(self):
        pass

    def __call__(self, inp, tar):
        inp = normalize_01(inp)

        return inp, tar


class Normalize:
    """Normalize based on mean and standard deviation."""

    def __init__(self,
                 mean: float,
                 std: float,
                 transform_input=True,
                 transform_target=False
                 ): 

        self.transform_input = transform_input
        self.transform_target = transform_target
        self.mean = mean
        self.std = std

    def __call__(self, inp, tar):
        inp = normalize(inp)

        return inp, tar


