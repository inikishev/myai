from collections import abc
import typing as T

import joblib
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from ..data import DS

MEAN = 33.3643
STD = 78.5836

normalize = v2.Normalize([MEAN], [STD])

def _download_and_create_mnist(root):
    train = MNIST(root, train=True, download = True)
    test = MNIST(root, train=False, download = True)

    loader = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32), normalize])

    ds = DS()
    ds.add_samples_(train, loader = loader)
    ds.add_samples_(test, loader = loader)

    return ds

def get_MNIST70000(path = r"D:\datasets\MNIST70000.joblib") -> DS[tuple[torch.Tensor, int]]:
    """Returns entire MNIST classification DS with 70,000 samples, z-normalized.
    Each sample is a tuple with (1x28x28) tensor and a class index."""
    return joblib.load(path)

def get_MNIST10000(path = r"D:\datasets\MNIST10000.joblib") -> DS[tuple[torch.Tensor, int]]:
    """Returns subset of MNIST classification DS with 1000 samples per each of 10 classes, z-normalized.
    Each sample is a tuple with (1x28x28) tensor and a class index."""
    return joblib.load(path)

def get_MNIST1000(path = r"D:\datasets\MNIST1000.joblib") -> DS[tuple[torch.Tensor, int]]:
    """Returns subset of MNIST classification DS with 100 samples per each of 10 classes, z-normalized.
    Each sample is a tuple with (1x28x28) tensor and a class index."""
    return joblib.load(path)

def get_MNIST100(path = r"D:\datasets\MNIST100.joblib") -> DS[tuple[torch.Tensor, int]]:
    """Returns subset of MNIST classification DS with 10 samples per each of 10 classes, z-normalized.
    Each sample is a tuple with (1x28x28) tensor and a class index."""
    return joblib.load(path)

def get_MNIST10(path = r"D:\datasets\MNIST10.joblib") -> DS[tuple[torch.Tensor, int]]:
    """Returns subset of MNIST classification DS with a single sample per each of 10 classes, z-normalized.
    Each sample is a tuple with (1x28x28) tensor and a class index."""
    return joblib.load(path)


def _make_even_subsampled_dataset(dataset: abc.Iterable[tuple[T.Any, T.Any]], samples_per_class: int):
    """Dataset must be an iterable of (sample, class). This creates a new DS with first samples_per_class samples per each class."""
    samples: dict[T.Any, list] = {}
    for img, label in dataset:
        if label in samples:
            if len(samples[label]) == samples_per_class: continue
            samples[label].append((img, label))
        else: samples[label] = [(img, label)]

    ds = DS()
    for v in samples.values():
        ds.add_samples_(v)
    return ds