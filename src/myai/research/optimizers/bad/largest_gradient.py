import hashlib

import torch
import torchzero as tz


class LargestGradient(tz.core.TensorListOptimizer):
    """only minimizes the parameter with the largest gradient norm, leading to hindered convergence"""
    def __init__(self, params, lr=1e-3, norm = 2):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.norm = norm

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params = self.get_params().ensure_grad_()
        grads = params.grad
        lr = self.get_group_key('lr')

        assert len(params) == len(lr)

        max_norm_idx = 0
        max_norm = 0
        norms = grads.norm(self.norm)
        # argmax
        for i,n in enumerate(norms):
            if n > max_norm:
                max_norm = n
                max_norm_idx = i

        params[max_norm_idx].sub_(grads[max_norm_idx] * lr[max_norm_idx])

        return loss