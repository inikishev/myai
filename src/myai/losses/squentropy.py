import typing as T
import warnings

import torch

from ..torch_tools import batched_one_hot_mask

class Squentropy(torch.nn.Module):
    def __init__(
        self,
        true_scale: float = 1,
        onehot_scale: float = 1,
        ignore_bg=False,
        mean_channels_batch = False,
        weight = None,
        add_spatial = False,
        ord = 2,
    ):
        """Hui, L., Belkin, M., & Wright, S. (2023, July). Cut your losses with squentropy. In International Conference on Machine Learning (pp. 14114-14131). PMLR.

        Cross-entropy loss plus average square loss over the incorrect classes.

        :param true_scale: Rescales loss at true label, defaults to 1
        :param onehot_scale: rescales the one-hot encoding, defaults to 1
        :param ce_act: activation to use for cross entropy.
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
        self.add_spatial = add_spatial
        self.ord = ord

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        input: `(B, C, *)`. Unnormalized logits without softmax or sigmoid.
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

        # check shapes
        if input.shape != target.shape:
            raise ValueError(f'{input.shape = } and {target.shape = }')

        # make sure target has correct dtype
        target = target.to(input.dtype, copy = False)

        # dims to sum over
        spatial_dims = list(range(2, input.ndim))
        has_spatial_dims = len(spatial_dims) > 0

        # cross-entropy
        if num_classes == 1:
            ce_loss = -(target * torch.nn.functional.logsigmoid(input) + (1 - target) * torch.nn.functional.logsigmoid(1 - input)) # pylint:disable = not-callable
        else:
            ce_loss = -torch.nn.functional.log_softmax(input, dim = 1) * target


        # multiply one-hot vector by M
        if self.onehot_scale != 1: target = target * self.onehot_scale

        # square loss (or other order)
        square_loss = input - target
        if self.ord % 2 != 0: square_loss = square_loss.abs()
        if self.ord != 1: square_loss = square_loss**self.ord

        # rescale loss at true label by k
        if self.true_scale != 1:
            true_mask = target == 1
            square_loss[true_mask] = square_loss[true_mask] * self.true_scale

        # add before averaging spatial dims
        if self.add_spatial:
            loss = square_loss + ce_loss
            if has_spatial_dims:
                loss = loss.mean(dim = spatial_dims)

        # else add after averaging spatial dims
        else:
            if has_spatial_dims:
                square_loss = square_loss.mean(dim = spatial_dims)
                ce_loss = ce_loss.mean(dim = spatial_dims)

            loss = square_loss + ce_loss

        # multiply loss by class weights, the loss is currently (B, C) so broadcastable into C
        if self.weight is not None:
            self.weight = self.weight.to(square_loss.device, copy=False)
            if square_loss.shape[-1] != len(self.weight):
                raise ValueError(f'Weights are {self.weight.shape} and dice is {square_loss.shape}')
            square_loss *= self.weight

        if self.mean_channels_batch: return square_loss.mean()
        else: return square_loss.mean(1).mean(0) # average over channels and then over batch
