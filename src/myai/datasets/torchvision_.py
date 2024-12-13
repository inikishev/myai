import os
from abc import ABC

import numpy as np
import torch
import torchvision.datasets
from torchvision.transforms import v2

from ..data import DS
from .base import ROOT, Dataset

class _TorchvisionClassificationDataset(Dataset):
    def __init__(self, name, cls):
        self.root = os.path.join(ROOT, name)
        if not os.path.exists(self.root): os.mkdir(self.root)
        self.name = name
        self.cls = cls

    def _make(self):
        # download
        train = self.cls(root = self.root, transform=v2.Compose([v2.ToTensor(), v2.ToDtype(torch.float32)]), download=True)
        test = self.cls(root = self.root, transform=v2.Compose([v2.ToTensor(), v2.ToDtype(torch.float32)]), download=True, train=False)

        # stack
        images = torch.stack([i[0] for i in train] + [i[0] for i in test])
        labels = torch.tensor([i[1] for i in train] + [i[1] for i in test], dtype=torch.float32)

        # znormalize
        images -= images.mean((0, 2,3), keepdim=True)
        images /= images.std((0, 2,3), keepdim=True)

        return images, labels

    def _save(self):
        images, labels = self._make()
        np.savez_compressed(os.path.join(self.root, f'{self.name}.npz'), images=images, labels = labels)

    def get(self) -> DS[tuple[torch.Tensor, int]]:
        data = np.load(os.path.join(self.root, f'{self.name}.npz'))
        images = data['images']
        labels = data['labels']
        ds = DS()
        ds.add_samples_([(torch.from_numpy(i), torch.tensor(l, dtype=torch.float32)) for i, l in zip(images, labels)])
        return ds


MNIST = _TorchvisionClassificationDataset('MNIST', torchvision.datasets.MNIST)
"""70,000 images 1×28×28, entire dataset is znormalized. Each image has integer label 0 to 9.
First 60,000 samples are commonly used as train set."""
CIFAR10 = _TorchvisionClassificationDataset('CIFAR10', torchvision.datasets.CIFAR10)
