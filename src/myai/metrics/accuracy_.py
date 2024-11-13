import torch

def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor):
    """
    Computes the classification accuracy.

    Parameters:
    y_pred (torch.Tensor): (B, C) Predicted labels, where C is the number of classes (softmax output).
    y_true (torch.Tensor): (B) or one-hot (B, C).

    Returns:
    torch.Tensor: Accuracy.
    """
    if y_true.ndim == 2:
        y_true = y_true.argmax(1)
    # Compute the number of correct predictions
    correct = torch.sum(y_pred.argmax(1) == y_true).to(torch.float32)

    # Compute the accuracy
    return correct / y_true.numel()