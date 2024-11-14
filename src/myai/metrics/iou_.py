import torch

from ..torch_tools import batched_one_hot_mask

def iou(y:torch.Tensor, yhat:torch.Tensor,):
    """
    Intersection over union metric often used for segmentation, also known as Jaccard index.

    y: ground truth in `(B, C, *)` or `(B, *)`.
    yhat: prediction in `(B, C, *)`
    reduce: `None`, `"mean"` or `"sum"`.

    returns: vector of len C with iou per each channel
    """
    # binary case, we convert it to 2 channels
    # so that we get separate dice for background and the class
    if yhat.shape[1] == 1:
        yhat = batched_one_hot_mask(y[:, 0], num_classes=yhat.shape[1])

    # make yhat (B, C, *), it is intended that this runs after previous line
    if yhat.ndim - y.ndim == 1:
        y = batched_one_hot_mask(y, num_classes=yhat.shape[1])

    # now that we processed y we also make sure it has at least 2 channels.
    if y.shape[1] == 1:
        y = batched_one_hot_mask(y[:, 0], num_classes=yhat.shape[1])

    y = y.to(torch.bool)
    yhat = yhat.to(torch.bool)

    intersection = (y & yhat).sum((0, *list(range(2, y.ndim))))
    union = (y | yhat).sum((0, *list(range(2, y.ndim))))
    return intersection / union