import torch
from ..torch_tools import batched_one_hot_mask
def _ensure_onehot(input: torch.Tensor, target: torch.Tensor):
    """Ensures input and target are one hot encoded boolean tensors.

    input (torch.Tensor): (B, C, *) Predicted labels, where C is the number of classes.
    target (torch.Tensor): (B, *) or one-hot (B, C, *).
    """
    if input.ndim - target.ndim not in (0, 1):
        raise ValueError(f'Invalid shapes, {input.shape = }; {target.shape = }. Target must have same or 1 less dims than input.')

    # binary case, we convert it to 2 channels
    # so that we get separate dice for background and the class
    if input.shape[1] == 1:
        input = batched_one_hot_mask(target[:, 0], num_classes=input.shape[1])

    # make yhat (B, C, *), it is intended that this runs after previous line
    if input.ndim - target.ndim == 1:
        target = batched_one_hot_mask(target, num_classes=input.shape[1])

    # now that we processed y we also make sure it has at least 2 channels.
    if target.shape[1] == 1:
        target = batched_one_hot_mask(target[:, 0], num_classes=input.shape[1])

    target = target.to(torch.bool)
    input = input.to(torch.bool)

    return input, target