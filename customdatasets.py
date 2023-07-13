import torch
from skimage.io import imread
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

class CustomDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            from multiprocessing import Pool
            from itertools import repeat

            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets, repeat(self.pre_transform)))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = imread(input_ID), imread(target_ID)
        y[y == 255] = 0

        
        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)
        

        
        x, y = torch.from_numpy(x.copy()).type(self.inputs_dtype), torch.from_numpy(y.copy()).type(self.targets_dtype)

        return x, y

    @staticmethod
    def read_images(inp, tar, pre_transform):
        inp, tar = imread(inp), imread(tar)
        if pre_transform:
            inp, tar = pre_transform(inp, tar)
        return inp, tar
