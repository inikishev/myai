from collections.abc import Sequence
from typing import Literal

import torch

from ..python_tools import reduce_dim
from .crop_ import crop as _crop


def _pad(
    input: torch.Tensor,
    padding: Sequence[int],
    mode: str = "constant",
    value=None,
    where: Literal["center", "start", "end"] = "center",
) -> torch.Tensor:
    # create padding sequence for torch.nn.functional.pad
    if where == 'center':
        torch_padding = [(int(i / 2), int(i / 2)) if i % 2 == 0 else (int(i / 2), int(i / 2) + 1) for i in padding]
    elif where == 'start':
        torch_padding = [(i, 0) for i in padding]
    elif where == 'end':
        torch_padding = [(0, i) for i in padding]
    else: raise ValueError(f'Invalid where: {where}')

    # broadcasting (e.g. if padding 3×128×128 by [4, 4], it will pad by [0, 4, 4])
    if len(torch_padding) < input.ndim:
        torch_padding = [(0, 0)] * (input.ndim - len(torch_padding)) + torch_padding

    if mode == 'zeros': mode = 'constant'; value = 0
    elif mode == 'min': mode = 'constant'; value = float(input.min())

    return torch.nn.functional.pad(input, reduce_dim(reversed(torch_padding)), mode=mode, value=value)

def pad(
    input: torch.Tensor,
    padding: Sequence[int],
    mode: str = "constant",
    value=None,
    where: Literal["center", "start", "end"] = "center",
) -> torch.Tensor:
    """
    Padding function that is easier to read:

    `output.shape[i]` = `input.shape[i] + padding[i]`.

    Args:
        input (torch.Tensor): input to pad.
        padding (str): How much padding to add per each dimension of `input`.
        mode (str, optional): Padding mode (https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html). Defaults to 'constant'.
        value (_type_, optional): Padding constant value. Defaults to None.
        where (str, optional): How to pad, if `center`, will pad start and end of each dimension evenly, if `start`, will pad at the start of each dimension , if `end`, will pad at the end. Defaults to 'center'.

    Returns:
        torch.Tensor: Padded `input`.
    """
    pad_values = [i if i > 0 else 0 for i in padding]
    if sum(pad_values) > 0:
        input = _pad(input, pad_values, mode, value, where)
    crop_values = [-i if i < 0 else 0 for i in padding]
    if sum(crop_values) > 0:
        input =  _crop(input, crop_values, where=where)
    return input

def pad_to_shape(
    input:torch.Tensor,
    shape:Sequence[int],
    mode:str = "constant",
    value=None,
    where:Literal["center", "start", "end"] = "center",
    crop = False,
) -> torch.Tensor:
    if crop: min = -99999999999999999999
    else: min = 0

    if len(shape) < input.ndim:
        shape = list(input.shape[:input.ndim - len(shape)]) + list(shape)

    return pad(
        input=input,
        padding=[max(min, shape[i] - input.shape[i]) for i in range(input.ndim)],
        mode=mode,
        value=value,
        where=where,
    )

def pad_like(input:torch.Tensor, target:torch.Tensor, where:Literal['center', 'start', 'end'] = 'center', mode='constant', value=None, crop = False,):
    return pad_to_shape(input, target.size(), where=where, mode=mode, value=value, crop=crop)


def pad_to_channels(input:torch.Tensor, channels:int, where:Literal['center', 'start', 'end'] = 'start', mode='constant', value=None, crop=False):
    """Pads (B, C, *) to (B, C2, *)

    Args:
        input (torch.Tensor): Input to pad in (B, C, *) format.
        target (torch.Tensor): Target that the input will match the shape of, must have same or bigger shape among all dimensions.
        where (str, optional): How to pad, if `center`, input will be in the center, etc. Defaults to 'center'.
        mode (str, optional): Padding mode from `torch.nn.functional.pad`, can be 'constant', 'reflect', 'replicate' or 'circular'. Defaults to 'constant'.
        value (_type_, optional): Constant padding value if `mode` is `constant`. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    return pad_to_shape(input, (input.shape[0], channels, *input.shape[2:]), where=where, mode=mode, value=value, crop = crop)



def pad_to_channels_like(input:torch.Tensor, target:torch.Tensor, where:Literal['center', 'start', 'end'] = 'center', mode='constant', value=None, crop=False):
    return pad_to_channels(input, target.size(1), where=where, mode=mode, value=value, crop=crop)



