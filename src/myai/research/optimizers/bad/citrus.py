import torch
from torch.optim import Optimizer

class Citrus(Optimizer):
    def __init__(self, params, lr=1e-3, peel_factor=0.1, eps=1e-8):
        defaults = dict(lr=lr, peel_factor=peel_factor, eps=eps)
        super(Citrus, self).__init__(params, defaults)
    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            peel_factor = group['peel_factor']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Citrus optimizer does not support sparse gradients')

                # Calculate the L2 norms for gradient and parameter
                grad_norm = torch.norm(grad, 2)
                param_norm = torch.norm(p.data, 2)

                # Compute the juice factor
                juice = grad_norm / (param_norm + eps)

                # Effective learning rate for this parameter
                effective_lr = lr * juice

                # Calculate the proposed update
                delta = effective_lr * grad

                # Compute the maximum allowed delta, adding eps to avoid zero clamping
                max_delta = peel_factor * (torch.abs(p.data) + eps)

                # Clip the delta to the range [-max_delta, max_delta]
                delta_clipped = torch.clamp(delta, -max_delta, max_delta)

                # Apply the clipped update
                p.sub_(delta_clipped)

        return loss