import os
from abc import ABC

import numpy as np
import torch
from torchvision.transforms import v2

from ..data import DS

ROOT = r'E:\datasets'
class Dataset(ABC):
    root: str

    def __init__(self):
        pass

    def _make(self, *args, **kwargs):
        """Stacks entire dataset into arrays. Downloads if needed/possible"""
        raise NotImplementedError

    def _save(self, *args, **kwargs):
        """Saves dataset to disk"""
        raise NotImplementedError

    def get(self, *args, **kwargs):
        """Returns a DS"""
        raise NotImplementedError

    def make_val_submission(self, outfile, model, *args, **kwargs):
        """Makes a submission file"""
        raise NotImplementedError
    
    def inference(self, model, input, *args, **kwargs):
        """run inference on some file, applies same transforms to it"""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)