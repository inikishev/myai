from typing import Any
from collections.abc import Sequence
from abc import ABC, abstractmethod

import torch


def _make_in_out_scale(in_x: Sequence[int] | None, out_x: Sequence[int] | None, x_scale: float | None) -> tuple[Sequence[int] | None, Sequence[int] | None, float | None]:
    # if all are None, scale is 1.
    if all(i is None for i in [in_x, out_x, x_scale]): x_scale = 1

    # all are defined, make sure they are consistent
    if all(i is not None for i in [in_x, out_x, x_scale]):
        in_x = [round(out_dim / x_scale) for out_dim in out_x] # type:ignore
        if any(i != in_x[0] for i in in_x):
            raise ValueError(f'Inconsistent {in_x = }, {out_x = }, {x_scale = }: {[round(out_dim / x_scale) for out_dim in out_x]}') # type:ignore

    if in_x is not None:

        # determine out_x from scale
        if out_x is None:
            if x_scale is None: x_scale = 1
            out_x = [round(in_dim * x_scale) for in_dim in in_x]

        # determine scale from out_x
        elif x_scale is None:
            scales = [o/i for i, o in zip(in_x, out_x)]
            if len(set(scales)) == 1: x_scale = scales[0]

    elif out_x is not None:

        # determine in_x from scale (in_x is always None there)
        if x_scale is not None:
            in_x = [round(out_dim / x_scale) for out_dim in out_x]

    return in_x, out_x, x_scale

def _make_x_shape_size_channels(x_shape: Sequence[int] | None, x_size: Sequence[int] | None, x_channels: int | None) -> tuple[Sequence[int] | None, Sequence[int] | None, int | None]:

    # x_shape from x_channels and x_size
    if x_shape is None:
        if (x_channels is not None) and (x_size is not None):
            x_shape = [x_channels] + list(x_size)

    # x_channels and x_size from x_shape
    if x_shape is not None:
        # check
        if x_size is not None:
            if x_shape[1:] != x_size: raise ValueError(f'{x_shape = } is inconcistent with {x_size = }')
        if x_channels is not None:
            if x_shape[0] != x_channels: raise ValueError(f'{x_shape = } is inconcistent with {x_channels = }')

        if x_channels is None: x_channels = x_shape[0]
        if (x_size is None) and (len(x_shape) > 1): x_size = x_shape[1:]

    return x_shape, x_size, x_channels

def _ensure_list_or_none(x: int | Sequence[int] | None):
    if x is None: return x
    if isinstance(x, int): return [x]
    return x

def _solve(
    in_channels: int | None = None,
    out_channels: int | None = None,
    channels_scale: float | None = None,
    in_size: Sequence[int] | None = None,
    out_size: Sequence[int] | None = None,
    size_scale: float | None = None,
    in_shape: Sequence[int] | None = None,
    out_shape: Sequence[int] | None = None,
    shape_scale: float | None = None,
    ndim: int = 2,
    out_ndim: int | None = None,
) -> dict[str, Any]:
    if out_ndim is None: out_ndim = ndim
    in_shape, out_shape, in_size, out_size = map(_ensure_list_or_none, [in_shape, out_shape, in_size, out_size])

    # iteratively solve this to fill all variables based on known variables
    # 2 iterations is technically enough but I do 4 to make sure
    for _ in range(4):

        in_shape, out_shape, shape_scale = _make_in_out_scale(in_shape, out_shape, shape_scale)
        in_size, out_size, size_scale = _make_in_out_scale(in_size, out_size, size_scale)

        inc, outc, channels_scale = _make_in_out_scale(_ensure_list_or_none(in_channels), _ensure_list_or_none(out_channels), channels_scale)
        if inc is not None: in_channels = inc[0]
        if outc is not None: out_channels = outc[0]

        in_shape, in_size, in_channels = _make_x_shape_size_channels(in_shape, in_size, in_channels)
        out_shape, out_size, out_channels = _make_x_shape_size_channels(out_shape, out_size, out_channels)


    return dict(
        in_channels = in_channels,
        out_channels = out_channels,
        channels_scale = channels_scale,
        in_size = in_size,
        out_size = out_size,
        size_scale = size_scale,
        in_shape = in_shape,
        out_shape = out_shape,
        shape_scale = shape_scale,
        ndim = ndim,
        out_ndim = out_ndim,
    )


def _check_can_change_x(in_x, out_x, x_scale, x_name, cls_name):
    if in_x != out_x or (x_scale is not None and x_scale != 1):
        raise RuntimeError(f"{cls_name} doesn't support changing number of {x_name}, got in_{x_name} = {in_x}, out_{x_name} = {out_x}, {x_name}_scale = {x_scale = }")

class Block(ABC):
    _can_change_channels = True
    _can_change_size = True
    _can_change_ndim = True

    def new(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        channels_scale: float | None = None,
        in_size: Sequence[int] | None = None,
        out_size: Sequence[int] | None = None,
        size_scale: float | None = None,
        in_shape: Sequence[int] | None = None,
        out_shape: Sequence[int] | None = None,
        shape_scale: float | None = None,
        ndim: int = 2,
        out_ndim: int | None = None,
    ):
        # check
        if not self._can_change_channels:
            _check_can_change_x(in_channels, out_channels, channels_scale, 'channels', self.__class__.__name__)

        if not self._can_change_size:
            _check_can_change_x(in_size, out_size, size_scale, 'size', self.__class__.__name__)

        if not self._can_change_ndim:
            _check_can_change_x(ndim, out_ndim, None, 'ndim', self.__class__.__name__)

        return self._make(
            **_solve(
                in_channels = in_channels,
                out_channels = out_channels,
                channels_scale = channels_scale,
                in_size = in_size,
                out_size = out_size,
                size_scale = size_scale,
                in_shape = in_shape,
                out_shape = out_shape,
                shape_scale = shape_scale,
                ndim = ndim,
                out_ndim = out_ndim,
            )
        )


    @abstractmethod
    def _make(
        self,
        in_shape: Sequence[int] | None,
        out_shape: Sequence[int] | None,
        shape_scale: float | None,
        in_size: Sequence[int] | None,
        out_size: Sequence[int] | None,
        size_scale: float | None,
        in_channels: int | None,
        out_channels: int | None,
        channels_scale: float | None,
        ndim: int,
        out_ndim: int,
    ) -> torch.nn.Module:
        ...

