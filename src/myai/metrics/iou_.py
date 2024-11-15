import torch

from ._utils import _ensure_onehot


def iou(input:torch.Tensor, target:torch.Tensor,):
    """
    Intersection over union metric often used for segmentation, also known as Jaccard index.

    input: prediction in `(B, C, *)`.
    target: ground truth in `(B, C, *)` or `(B, *)`.
    reduce: `None`, `"mean"` or `"sum"`.

    returns: vector of len C with iou per each channel
    """
    input, target = _ensure_onehot(input, target)

    intersection = (input & target).sum(list(range(2, target.ndim)))
    union = (input | target).sum(list(range(2, target.ndim)))
    
    return (intersection / union).nanmean(0) # mean along batch dim but not channel dim