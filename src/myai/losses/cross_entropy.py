from typing import Optional

import torch


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """Finally, after months of work, scientists have been able obtain the Cross Entropy Loss with Automatic Casting."""
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,):
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, target.to(input.dtype, copy = False))