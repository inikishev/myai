# pylint: disable=undefined-variable
# because its bugged with generics
import concurrent.futures
import operator
import typing as T
from collections import UserList, abc

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..python_tools import (Composable, SupportsIter, compose, func2method,
                            maybe_compose)
from ..rng import RNG


class Sample:
    def __init__(self, data, loader: Composable | None, transform: Composable | None):
        self.data = data
        self.loader: abc.Callable = maybe_compose(loader)
        self.transform = maybe_compose(transform)

        self.preloaded = None

    def __call__(self):
        if self.preloaded is not None: return self.transform(self.preloaded)
        return self.transform(self.loader(self.data))

    def preload_(self):
        if self.preloaded is None: self.preloaded = self.loader(self.data)

    def unload_(self):
        self.preloaded = None

    def add_loader_(self, loader: Composable):
        self.loader = compose(self.loader, loader)
    def set_loader_(self, loader: Composable | None):
        self.loader = maybe_compose(loader)

    def add_transform_(self, transform: Composable):
        self.transform = compose(self.transform, transform)
    def set_transform_(self, transform: Composable | None):
        self.transform = maybe_compose(transform)

    def copy(self):
        sample = self.__class__(self.data, self.loader, self.transform)
        sample.preloaded = self.preloaded
        return sample

class DS[R](abc.Sequence[R]):
    def __init__(self, n_threads = 0):
        super().__init__()
        self.samples: list[Sample] = []
        self.idxs: list[int] = []

        self.n_threads = n_threads
        if n_threads > 0: self._executor = concurrent.futures.ThreadPoolExecutor(n_threads)
        else: self._executor = None

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __getitem__(self, idx) -> R: # type:ignore
        return self.samples[self.idxs[idx]]()

    def __getitems__(self, indexes: abc.Iterable[int]) -> list[R]:
        if self._executor is not None:
            return list(self._executor.map(lambda i: self.samples[self.idxs[i]](), indexes))

        return [self.samples[self.idxs[i]]() for i in indexes]

    def __len__(self):
        return len(self.samples)

    def _add_sample_object(self, sample: Sample):
        self.idxs.append(len(self.samples))
        self.samples.append(sample)

    def _add_sample_objects(self, samples: list[Sample]):
        self.idxs.extend(range(len(self.samples), len(self.samples) + len(samples)))
        self.samples.extend(samples)

    def add_sample_(self, data, loader: Composable | None = None, transform: Composable | None = None):
        self._add_sample_object(Sample(data, loader, transform))

    def add_samples_(self, samples: SupportsIter, loader: Composable | None = None, transform: Composable | None = None):
        self._add_sample_objects([Sample(s, loader, transform) for s in samples]) #type:ignore

    def merge_(self, ds: "DS"):
        self._add_sample_objects(ds.samples)

    def merged_with(self, ds: "DS"):
        merged = self.__class__(n_threads = self.n_threads)
        merged._add_sample_objects(self.samples)
        merged._add_sample_objects(ds.samples)
        return merged

    def copy(self, copy_samples = True):
        ds = self.__class__(n_threads = self.n_threads)

        if copy_samples: samples = [i.copy() for i in self.samples]
        else: samples = self.samples.copy()

        ds._add_sample_objects(samples)
        return ds

    def dataloader(self, batch_size: int = 1, shuffle = False, ):
        return DataLoader(self, batch_size = batch_size, shuffle = shuffle) # type:ignore

    def add_loader_(self, loader: Composable):
        for s in self.samples: s.add_loader_(loader)

    def set_loader_(self, loader: Composable | None):
        for s in self.samples: s.set_loader_(loader)

    def add_transform_(self, transform: Composable):
        for s in self.samples: s.add_transform_(transform)

    def set_transform_(self, transform: Composable | None):
        for s in self.samples: s.set_transform_(transform)

    def shuffle_(self, seed = None):
        RNG(seed).random.shuffle(self.idxs)

    def _ensure_absolute_amount(self, amount: int | float | None):
        if amount is None: return len(self)
        if isinstance(amount, float): return int(amount * len(self))
        return amount

    def preload_(self, amount: int | float | None = None, clear_data = False):
        """Preloads all or first `amount` samples."""
        amount = self._ensure_absolute_amount(amount)

        if self._executor is not None:
            with self._executor as ex:
                for _ in ex.map(operator.methodcaller('preload'), self.samples[:amount]): pass
        else:
            for s in self.samples[:amount]: s.preload_()

        if clear_data:
            for s in self.samples[:amount]: s.data = None

    def split(self, splits: int | float | abc.Sequence[int | float], shuffle = True, seed = None) -> "list[DS[R]]":
        if isinstance(splits, (int, float)): splits = [splits, ]

        splits = [self._ensure_absolute_amount(s) for s in splits]
        if len(splits) == 1: splits.append(len(self) - splits[0])

        idxs = list(range(len(self.samples)))
        if shuffle:
            RNG(seed).random.shuffle(idxs)

        datasets = [DS(self.n_threads) for _ in splits]

        cur = 0
        for i, s in enumerate(splits):
            datasets[i]._add_sample_objects([self.samples[i] for i in idxs[cur:cur+s]])
            cur += s

        return datasets

    def calculate_mean_std(
        self,
        dim: int | list[int] | tuple[int] | None = 0,
        batch_size = 32,
    ):
        if dim is None: dim = []
        elif isinstance(dim, int): dim = [dim]

        dl = torch.utils.data.DataLoader(self, batch_size=batch_size,) # type:ignore

        mean = std = None
        nsamples = 0
        for sample in dl:
            # get the actual sample if there is a label
            if isinstance(sample, (list, tuple)):  sample = sample[0]
            sample: torch.Tensor

            # find the reduction dims
            # d + 1 because we have additional batch dim
            other_dims = [d for d in range(sample.ndim) if d+1 not in dim]

            # create mean and std counters
            if mean is None:
                if len(dim) == 0:
                    mean = 0; std = 0
                else:
                    mean = torch.zeros([sample.shape[d+1] for d in dim])
                    std = torch.zeros([sample.shape[d+1] for d in dim])


            mean += sample.mean(dim=other_dims)
            std += sample.std(dim=other_dims)
            nsamples += 1

        if mean is None or std is None: raise ValueError('Dataset is empty.')
        return mean / nsamples, std / nsamples


