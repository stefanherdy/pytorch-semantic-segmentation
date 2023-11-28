#!/usr/bin/env python3
# Detailed description about image augmentation and how to use the code:
# https://medium.com/@stefan.herdy/how-to-augment-images-for-semantic-segmentation-2d7df97544de

import numpy as np
from sklearn.externals._pilutil import bytescale
import random
import matplotlib.pyplot as plt
import cv2
import random
import torch
import torchvision.transforms as transforms

def create_dense_target(tar: np.ndarray):
    dense_tar = tar[:,:]
    return dense_tar

def normalize_01(inp: np.ndarray):
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
    Composes several transformations together.
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
        
        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class ColorTransformations:
    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        inp_tensor = torch.from_numpy(inp)
        tar_tensor = torch.from_numpy(tar)

        color_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ])

        inp_tensor = color_transform(inp_tensor)

        inp = inp_tensor.numpy()
        tar = tar_tensor.numpy()

        return inp, tar

class ColorNoise:
    def __init__(self, noise_std=0.05):
        self.noise_std = noise_std

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        inp_tensor = torch.from_numpy(inp)
        tar_tensor = torch.from_numpy(tar)

        noise = torch.randn_like(inp_tensor) * self.noise_std
        inp_tensor += noise

        inp_tensor = torch.clamp(inp_tensor, 0, 1)

        inp = inp_tensor.numpy()
        tar = tar_tensor.numpy()

        return inp, tar

class RandomFlip:

    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        rand = random.choice([0, 1])
        if rand == 1:
            inp = np.moveaxis(inp, 0, -1)
            inp = cv2.flip(inp, 1)
            inp = np.moveaxis(inp, -1, 0)
            tar = np.ndarray.copy(np.fliplr(tar))

        rand = random.choice([0, 1])
        if rand == 1:
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

    def __init__(self, crop_size):
        self.crop_size = crop_size
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        max_x = inp.shape[1] - self.crop_size
        max_y = inp.shape[2] - self.crop_size
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        inp = np.moveaxis(inp, 0, -1)
        inp = inp[x: x + self.crop_size, y: y + self.crop_size,:]
        inp = np.moveaxis(inp, -1, 0)
        tar = tar[x: x + self.crop_size, y: y + self.crop_size]

        return inp, tar


class Resize:

    def __init__(self, img_size):
        self.img_size = img_size
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        inp = np.moveaxis(inp, 0, -1)
        inp = cv2.resize(inp, (self.img_size,self.img_size), interpolation = cv2.INTER_NEAREST)
        inp = np.moveaxis(inp, -1, 0)
        tar = cv2.resize(tar, (self.img_size,self.img_size), interpolation = cv2.INTER_NEAREST)
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


