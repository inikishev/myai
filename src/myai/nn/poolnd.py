import typing
from collections.abc import Sequence, Callable

import torch

class PoolLike(typing.Protocol):
    """Protocol for pooling classes."""
    def __call__(
        self,
        kernel_size,
        stride: typing.Any = None,
        padding: typing.Any = 0,
        dilation: typing.Any = 1,
        ndim: int = 2,
    ) -> Callable[[torch.Tensor], torch.Tensor]: ...

def _get_maxpoolnd_cls(ndim: int,):
    """Returns a class."""
    if ndim == 1: return torch.nn.MaxPool1d
    elif ndim == 2: return torch.nn.MaxPool2d
    elif ndim == 3: return torch.nn.MaxPool3d
    else: raise ValueError(f'Invalid ndim {ndim}.')

def maxpoolnd(
    kernel_size,
    stride: typing.Any = None,
    padding: typing.Any = 0,
    dilation: typing.Any = 1,
    ndim = 2,
):
    kwargs = locals().copy()
    del kwargs['ndim']
    return _get_maxpoolnd_cls(ndim)(**kwargs)

__test_maxpoolnd: PoolLike = maxpoolnd

def _get_avgpoolnd_cls(ndim: int,):
    """Returns a class."""
    if ndim == 1: return torch.nn.AvgPool1d
    elif ndim == 2: return torch.nn.AvgPool2d
    elif ndim == 3: return torch.nn.AvgPool3d
    else: raise ValueError(f'Invalid ndim {ndim}.')


def avgpoolnd(
    kernel_size,
    stride: typing.Any = None,
    padding: typing.Any = 0,
    dilation = 1,
    ndim = 2,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
):
    kwargs = locals().copy()
    del kwargs['ndim']
    return _get_maxpoolnd_cls(ndim)(**kwargs)

__test_mavgpoolnd: PoolLike = avgpoolnd
