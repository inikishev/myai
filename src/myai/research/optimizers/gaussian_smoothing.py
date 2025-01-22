import torch
import copy

class GaussianHomotopy(torch.optim.Optimizer):
    """(sort of) gaussian homotopy, I just made sigma decay slowly."""
    def __init__(self, params, lr=1e-3, num_samples=5, sigma=1., sigma_decay = 0.99):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if num_samples <= 0:
            raise ValueError(f"Invalid num_samples: {num_samples}")
        if sigma < 0.0:
            raise ValueError(f"Invalid perturb_scale: {sigma}")

        defaults = dict(lr=lr, num_samples=num_samples, sigma=sigma, sigma_decay=sigma_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required for PotentialGradientOptimizer")

        # Save original parameters
        original_params = [
            { 'params': [p.detach().clone() for p in group['params']] }
            for group in self.param_groups
        ]

        # Prepare for gradient accumulation
        avg_gradients = None
        total_samples = self.defaults['num_samples']
        sigma = self.defaults['sigma']

        for _ in range(total_samples):
            # Perturb parameters with Gaussian noise
            for group, orig in zip(self.param_groups, original_params):
                for p, orig_p in zip(group['params'], orig['params']):
                    noise = torch.randn_like(p) * sigma
                    p.data.add_(noise)

            # Compute loss and gradients for perturbed parameters
            with torch.enable_grad(): loss = closure()
            #loss.backward()

            # Accumulate gradients
            if avg_gradients is None:
                avg_gradients = [
                    [p.grad.clone() for p in group['params']]
                    for group in self.param_groups
                ]
            else:
                for group_idx, group in enumerate(self.param_groups):
                    for p_idx, p in enumerate(group['params']):
                        avg_gradients[group_idx][p_idx].add_(p.grad)

            # Restore original parameters and zero gradients
            for group, orig in zip(self.param_groups, original_params):
                for p, orig_p in zip(group['params'], orig['params']):
                    p.data.copy_(orig_p)
                    if p.grad is not None:
                        p.grad.zero_()

        # Average gradients and update parameters
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr'] if 'lr' in group else self.defaults['lr']
            for p_idx, p in enumerate(group['params']):
                avg_grad = avg_gradients[group_idx][p_idx] / total_samples # type:ignore
                if p.grad is None:
                    p.grad = avg_grad
                else:
                    p.grad.copy_(avg_grad)
                p.data.add_(p.grad, alpha=-lr)

        self.defaults['sigma'] *= self.defaults['sigma_decay']
        return loss # type:ignore