import typing as T
from collections import abc

import torch


class PoolLike(T.Protocol):
    """Protocol for pooling classes."""
    def __call__(
        self,
        kernel_size,
        stride: T.Any = None,
        padding: T.Any = 0,
        dilation: T.Any = 1,
        ndim: int = 2,
    ) -> abc.Callable[[torch.Tensor], torch.Tensor]: ...

def _get_maxpoolnd_cls(ndim: int,):
    """Returns a class."""
    if ndim == 1: return torch.nn.MaxPool1d
    elif ndim == 2: return torch.nn.MaxPool2d
    elif ndim == 3: return torch.nn.MaxPool3d
    else: raise ValueError(f'Invalid ndim {ndim}.')

def maxpoolnd(
    kernel_size,
    stride: T.Any = None,
    padding: T.Any = 0,
    dilation: T.Any = 1,
    ndim = 2,
):
    kwargs = locals().copy()
    del kwargs['ndim']
    return _get_maxpoolnd_cls(ndim)(**kwargs)

__test_maxpoolnd: PoolLike = maxpoolnd

