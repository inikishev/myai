from typing import Any, overload
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
    ndim: int = 2,
    out_ndim: int | None = None,
) -> dict[str, Any]:
    if out_ndim is None: out_ndim = ndim

    # iteratively solve this to fill all variables based on known variables
    # 2 iterations is technically enough but I do 4 to make sure
    for _ in range(4):

        in_size, out_size, size_scale = _make_in_out_scale(in_size, out_size, size_scale)

        inc, outc, channels_scale = _make_in_out_scale(_ensure_list_or_none(in_channels), _ensure_list_or_none(out_channels), channels_scale)
        if inc is not None: in_channels = inc[0]
        if outc is not None: out_channels = outc[0]


    return dict(
        in_channels = in_channels,
        out_channels = out_channels,
        channels_scale = channels_scale,
        in_size = in_size,
        out_size = out_size,
        size_scale = size_scale,
        ndim = ndim,
        out_ndim = out_ndim,
    )


class Config:
    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        channels_scale: float | None = None,
        in_size: Sequence[int] | None = None,
        out_size: Sequence[int] | None = None,
        size_scale: float | None = None,
        ndim: int = 2,
        out_ndim: int | None = None,
        dtype=None,
        **extra,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels_scale = channels_scale
        self.in_size = in_size
        self.out_size = out_size
        self.size_scale = size_scale
        self.ndim = ndim
        self.out_ndim = out_ndim
        self.dtype = dtype
        self.extra = extra

    def __repr__(self):
        return f"Config(in_channels = {self.in_channels}, out_channels = {self.out_channels}, channels_scale = {self.channels_scale}, in_size = {self.in_size}, out_size = {self.out_size}, size_scale = {self.size_scale}, ndim = {self.ndim}, out_ndim = {self.out_ndim}, dtype = {self.dtype}, kwargs = {self.extra})"

    def kwargs(self, channels: bool, size: bool, other: str | Sequence[str] = ()) -> dict[str, Any]:
        k = {}
        if channels: k.update(dict(in_channels=self.in_channels, out_channels=self.out_channels, channels_scale=self.channels_scale))
        if size: k.update(dict(in_size=self.in_size, out_size=self.out_size, size_scale=self.size_scale))

        if isinstance(other, str): other = [other, ]
        for attr in other:
            k[attr] = getattr(self, attr)
        return k

    def all_kwargs(self, blacklist: str | Sequence[str] = ()) -> dict[str, Any]:
        if isinstance(blacklist, str): blacklist = [blacklist, ]

        kwargs = self.kwargs(True, True, ('ndim', 'out_ndim', 'dtype', 'extra'))
        kwargs = {k:v for k,v in kwargs if k not in blacklist}
        return kwargs

class Block(ABC):
    """A generic block.

    .. code::py

        conv = B.ExampleConv(bias=False)
        channels = [16,32,64,128]
        convs = [conv.new(x,y, size_scale=0.5) for x,y in zip(channels[:-1], channels[1:])]

    """
    def new(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        channels_scale: float | None = None,
        in_size: Sequence[int] | None = None,
        out_size: Sequence[int] | None = None,
        size_scale: float | None = None,
        ndim: int = 2,
        out_ndim: int | None = None,
        dtype = None,
        extra = None,
    ) -> torch.nn.Module:
        if extra is None: extra = {}

        c = Config(
            **_solve(
                in_channels=in_channels,
                out_channels=out_channels,
                channels_scale=channels_scale,
                in_size=in_size,
                out_size=out_size,
                size_scale=size_scale,
                ndim=ndim,
                out_ndim=out_ndim,
                **extra,
            ),
            dtype=dtype,
        )

        self._validate(c)
        return self._make(c)

    def from_config(self, c:Config): return self.new(**c.all_kwargs())

    def _validate(self, c: Config) -> None:
        pass

    @abstractmethod
    def _make(self, c: Config) -> torch.nn.Module:
        ...

class ChannelBlock(Block, ABC):
    """block that can change number of channels."""
    def new(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        channels_scale: float | None = None,
        dtype = None,
        extra = None,
    ):
        return super().new(in_channels=in_channels, out_channels=out_channels, channels_scale=channels_scale, dtype=dtype, extra=extra)

    def from_config(self, c:Config):
        return self.new(**c.kwargs(channels=True, size=False, other = ['dtype', 'extra']))

class SizeBlock(Block, ABC):
    """block that can upscale or downscale."""
    def new(
        self,
        size_scale: float | None = None,
        in_size: Sequence[int] | None = None,
        out_size: Sequence[int] | None = None,
        dtype = None,
        extra = None,
    ):
        return super().new(size_scale=size_scale, in_size=in_size, out_size=out_size, dtype=dtype, extra=extra)

    def from_config(self, c:Config):
        return self.new(**c.kwargs(channels=False, size=True, other = ['dtype', 'extra']))


class DownscaleBlock(SizeBlock, ABC):
    """block that can downscale, checks that scale <= 1"""
    def new(
        self,
        size_scale: float | None = None,
        in_size: Sequence[int] | None = None,
        out_size: Sequence[int] | None = None,
        dtype = None,
        extra = None,
    ):
        return super().new(size_scale=size_scale, in_size=in_size, out_size=out_size, dtype=dtype, extra=extra)

    def from_config(self, c:Config):
        return self.new(**c.kwargs(channels=False, size=True, other = ['dtype', 'extra']))

    def _validate(self, c):
        if c.size_scale is not None and c.size_scale > 1:
            raise ValueError(f"{self.__class__.__name__} doesn't support upsampling, got {c}")

class UpscaleBlock(SizeBlock, ABC):
    """block that can upscale, checks that scale >= 1"""
    def new(
        self,
        size_scale: float | None = None,
        in_size: Sequence[int] | None = None,
        out_size: Sequence[int] | None = None,
        dtype = None,
        extra = None,
    ):
        return super().new(size_scale=size_scale, in_size=in_size, out_size=out_size, dtype=dtype, extra=extra)

    def from_config(self, c:Config):
        return self.new(**c.kwargs(channels=False, size=True, other = ['dtype', 'extra']))

    def _validate(self, c):
        if c.size_scale is not None and c.size_scale < 1:
            raise ValueError(f"{self.__class__.__name__} doesn't support upsampling, got {c}")

class StraightBlock(Block, ABC):
    """block that doesn't change the shape of the input"""
    def new(self, dtype = None, extra = None):
        return super().new(dtype = dtype, extra = extra)

    def from_config(self, c:Config):
        return self.new(dtype=c.dtype, extra=c.extra)

class DebugPrint(Block):
    """prints config"""
    def _make(self, c):
        print(c)
        return torch.nn.Identity()

class DebugReturnConfig(Block):
    """`new` method returns config"""
    def _make(self, c):
        return c

class ExampleLinear(ChannelBlock):
    """Simplified example of a linear block"""
    def __init__(self, bias=True):
        self.bias = bias

    def _make(self, c):
        assert (c.in_channels is not None) and (c.out_channels is not None)
        return torch.nn.Linear(c.in_channels, c.out_channels, bias=self.bias, dtype=c.dtype)


class ExampleConv3(Block):
    """Simplified example of a convolutional block with 3x3 kernel"""
    def _make(self, c):
        assert c.ndim == 2 and c.out_ndim == 2
        assert c.size_scale in (1, 0.5)
        assert (c.in_channels is not None) and (c.out_channels is not None)

        return torch.nn.Conv2d(c.in_channels, c.out_channels, kernel_size=3, stride = 2 if c.size_scale == 0.5 else 1, padding = 1, dtype = c.dtype)



class ExampleChannelSizeChain(Block):
    """simplified example that chains channel block and size block into a full block"""
    def __init__(self, channel_block: ChannelBlock, size_block: SizeBlock):
        self.channel_block = channel_block
        self.size_block = size_block

    def _make(self, c):
        return torch.nn.Sequential(
            self.channel_block.from_config(c),
            self.size_block.from_config(c)
        )



def test_solve():
    block = DebugReturnConfig()
    cfg = block.new(10, 20)
    assert cfg.channels_scale == 2, cfg

    cfg = block.new(10, channels_scale=0.5)
    assert cfg.out_channels == 5, cfg

    cfg = block.new(in_size = (12, 12), out_size = (6, 6))
    assert cfg.size_scale == 0.5, cfg

    cfg = block.new(in_size = (12, 12), size_scale=2)
    assert cfg.out_size == [24, 24], cfg


if __name__ == '__main__':
    test_solve()