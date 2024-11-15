import torch
from ._utils import _ensure_onehot

def dice(input:torch.Tensor, target:torch.Tensor, ):
    """
    Sørensen–Dice coefficient often used for segmentation. Defined as two intersections over sum. Equivalent to F1 score.

    input: prediction in `(B, C, *)`.
    target: ground truth in `(B, C, *)` or `(B, *)`.
    reduce: `None`, `"mean"` or `"sum"`.

    returns: vector of len C with dice per each channel.
    """
    input, target = _ensure_onehot(input, target)

    intersection = (target & input).sum(list(range(2, target.ndim)))
    sum = target.sum(list(range(2, target.ndim))) + input.sum(list(range(2, target.ndim)))

    return ((2*intersection) / sum).nanmean(0) # mean along batch dim but not channel dim