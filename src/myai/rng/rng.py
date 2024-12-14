import random

import numpy as np
import torch


class RNG:
    def __init__(self, seed: int | None):
        self.seed = seed
        self.random = random.Random(seed)
        self.numpy = np.random.default_rng(seed)

        if seed is not None: self.torch = torch.Generator().manual_seed(seed)
        else: self.torch = None

    def copy(self):
        return RNG(self.seed)
