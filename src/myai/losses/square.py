import typing as T
import warnings

import torch

from ..torch_tools import batched_one_hot_mask


class RescaledSquareLoss(torch.nn.Module):
    def __init__(
        self,
        true_scale: float = 1,
        onehot_scale: float = 1,
        ignore_bg=False,
        mean_channels_batch = False,
        weight = None,
        ord = 2,
    ):
        """Hui, L., & Belkin, M. (2020). Evaluation of neural architectures trained with square loss vs cross-entropy in classification tasks. arXiv preprint arXiv:2006.07322.

        With k = M = 1 this is standard square loss, or brier score, or class-wise MSE.
        For multidimensional inputs the loss is averaged over spatial dimensions first.

        with ~50+ clases k and M can be increased, in the paper k = 1, M = 15 for 42-1000 classes, or K = 15 and M = 30 for 1000 classes.

        :param true_scale: Rescales loss at true label, defaults to 1
        :param onehot_scale: rescales the one-hot encoding, defaults to 1
        :param ignore_bg: Whether to ignore background, defaults to False
        :param mean_channels_batch: _description_, defaults to False
        :param weight: Weights of ecah class, defaults to None
        """
        super().__init__()
        self.ignore_bg = ignore_bg
        if weight is not None and not isinstance(weight, torch.Tensor): weight = torch.as_tensor(weight, dtype=torch.float32)
        self.weight = weight
        self.mean_channels_batch = mean_channels_batch
        self.true_scale = true_scale
        self.onehot_scale = onehot_scale
        self.ord = ord

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        input: `(B, C, *)`. Usually unnormalized logits, without softmax or sigmoid.
        target: `(B, C, *)` or `(B, *)`
        """
        num_classes = input.shape[1]

        # make sure target is one hot encoded
        if input.ndim - target.ndim == 1:

            # non-binary case
            if num_classes > 1:
                target = batched_one_hot_mask(target, num_classes = input.shape[1])

            # in binary case target is already 0s and 1s, we add the channel dimension to it.
            else:
                target = target.unsqueeze(1)

        # remove background channel
        if self.ignore_bg:
            input = input[:,1:]
            target = target[:, 1:]

        # chck shapes
        if input.shape != target.shape:
            raise ValueError(f'{input.shape = } and {target.shape = }')

        # make sure target has correct dtype
        target = target.to(input.dtype, copy = False)

        # dims to sum over
        spatial_dims = list(range(2, input.ndim))
        has_spatial_dims = len(spatial_dims) > 0

        # multiply one-hot vector by M
        if self.onehot_scale != 1: target = target * self.onehot_scale

        # square loss (or other order)
        loss = input - target
        if self.ord % 2 != 0: loss = loss.abs()
        if self.ord != 1: loss = loss**self.ord

        # rescale loss at true label by k
        if self.true_scale != 1:
            true_mask = target == 1
            loss[true_mask] = loss[true_mask] * self.true_scale

        # sum over spatial dimensions
        if has_spatial_dims: loss = loss.mean(dim = spatial_dims)

        # multiply loss by class weights, the loss is currently (B, C) so broadcastable into C
        if self.weight is not None:
            self.weight = self.weight.to(loss.device, copy=False)
            if loss.shape[-1] != len(self.weight):
                raise ValueError(f'Weights are {self.weight.shape} and dice is {loss.shape}')
            loss *= self.weight

        if self.mean_channels_batch: return loss.mean()
        else: return loss.mean(1).mean(0) # average over channels and then over batch
