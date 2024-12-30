import random

import torch

from ._base import RandomTransform

__all__ = [
    "add_gaussian_noise",
    "GaussianNoise",
    "add_gaussian_noise_triangular",
    "GaussianNoiseTriangular",
]
def add_gaussian_noise(x):
    return x + torch.randn_like(x)

class GaussianNoise(RandomTransform):
    """GAUSSIAN NOISE WILL NOT BE AFFECTED BY SEED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
    def __init__(self, p = 0.1, seed = None):
        super().__init__(seed)
        self.p = p
    def forward(self, x): return add_gaussian_noise(x)

def add_gaussian_noise_triangular(x):
    return x + torch.randn_like(x) * random.triangular(0, 1, 0)

class GaussianNoiseTriangular(RandomTransform):
    """GAUSSIAN NOISE WILL NOT BE AFFECTED BY SEED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
    def __init__(self, p = 0.1, seed = None):
        super().__init__(seed)
        self.p = p
    def forward(self, x): return add_gaussian_noise_triangular(x)
