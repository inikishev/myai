import torch
from ._utils import _ensure_onehot

def accuracy(input:torch.Tensor, target:torch.Tensor, ):
    """
    Computes accuracy.

    Parameters:
    input (torch.Tensor): (B, C, *) Predicted labels, where C is the number of classes.
    target (torch.Tensor): (B, *) or one-hot (B, C, *).

    Returns:
    torch.Tensor: Accuracy.
    """
    # argmax target if not argmaxed (input is always one hot encoded)
    if input.ndim == target.ndim:
        target = target.argmax(1)

    # Compute the number of correct predictions
    correct = torch.sum(input.argmax(1) == target, dim = list(range(1, target.ndim))).float()

    # Compute the accuracy
    return (correct / target[0].numel()).mean()