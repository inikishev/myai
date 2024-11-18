from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from torchzero.optim.core import OptimizerModule
from torchzero.optim.modules.operators.normalization import normalize_grad_
from torchzero.optim.modules.operators.sign import sign_grad_
from torchzero.optim.modules.smoothing.laplacian_smoothing import (
    LaplacianSmoothing as TorchzeroLaplacianSmoothing, gradient_laplacian_smoothing_)

from ...event_model import Callback

if TYPE_CHECKING:
    from ..learner import Learner


class GradClipNorm(Callback):
    """Gradient clipping by norm."""

    def __init__(self, max_norm: float, norm_type: float = 2,):
        super().__init__()
        self.max_norm = max_norm
        self.norm_type = norm_type

    def after_backward(self, learner: "Learner"):
        torch.nn.utils.clip_grad_norm_(learner.model.parameters(), self.max_norm, norm_type=self.norm_type)


class GradClipValue(Callback):
    """Gradient clipping by value."""

    def __init__(self, max_value: float):
        super().__init__()
        self.max_value = max_value

    def after_backward(self, learner: "Learner"):
        torch.nn.utils.clip_grad_value_(learner.model.parameters(), self.max_value)

class GradNorm(Callback):
    """Gradient normalization."""
    def __init__(self, norm_value: float, norm_type: float = 2, min:float = 0):
        super().__init__()
        self.norm_value = norm_value
        self.norm_type = norm_type
        self.min = min

    def after_backward(self, learner: "Learner"):
        normalize_grad_(learner.model.parameters(), self.norm_value, min = self.min, ord = self.norm_type)


class GradSign(Callback):
    """Takes the sign of the gradient."""

    def after_backward(self, learner: "Learner"):
        sign_grad_(learner.model.parameters())

class LaplacianSmoothing(Callback):
    """Applies laplacian smoothing to the gradient."""
    def __init__(self, sigma: float = 1, layerwise: bool = True):
        super().__init__()
        # we create a smoother because it will cache the denominator which will be faster
        self.smoother = TorchzeroLaplacianSmoothing(sigma = sigma, layerwise=layerwise)

    def after_backward(self, learner: "Learner"):
        if not self.smoother._initialized: self.smoother._initialize_(learner.model.parameters())
        grads = self.smoother.get_params().get_existing_grads()
        smooth_grads = self.smoother._update(None, grads)
        grads.set_(smooth_grads)

# class TorchzeroModule(Callback):
#     """Gradient normalization."""
#     def __init__(self, modules: OptimizerModule | Iterable[OptimizerModule]):
#         super().__init__()
#         if isinstance(modules, OptimizerModule): modules = [modules]
#         self.modules = list(modules)

#     def before_fit(self, learner: "Learner"):
#         for m in self.modules:
#             m.set_params_(learner.model.parameters())

#     def after_backward(self, learner: "Learner"):
#         params = self.modules[0].get_params()

#         closure = learner.make_closure(learner.batch)
#         fx0 = learner.loss
#         for module in self.modules:
#             module.update_ascent_direction_(closure, params.grad, fx0 = fx0)
#             if module.fx0 is not None:
#                 fx0 = module.fx0