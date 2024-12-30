import random
from abc import ABC, abstractmethod
from collections.abc import Callable

from ..rng import RNG


class Transform(ABC):

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError(self.__class__.__name__ + " doesn't have `transform` method.")

    def reverse(self, x):
        raise NotImplementedError(self.__class__.__name__ + " doesn't have `reverse` method.")

    def __call__(self, x):
        return self.forward(x)

class RandomTransform(Transform, ABC):
    p:float

    def __init__(self, seed: int | None = None):
        self.rng = RNG(seed)

    def __call__(self, x):
        if self.rng.random.random() < self.p: return self.forward(x)
        return x
