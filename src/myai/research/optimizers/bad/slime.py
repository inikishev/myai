import torch
from torch.optim import Optimizer
import math

class SlimeMoldVeinOptimizer(Optimizer):
    """directional consistency sounded better than it is"""
    def __init__(self, params, lr=1e-3, growth_rate=0.1, decay=0.99, min_strength=0.1, max_strength=10.0):
        defaults = dict(lr=lr, growth_rate=growth_rate, decay=decay,
                        min_strength=min_strength, max_strength=max_strength)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['previous_grad'] = torch.zeros_like(p.data)
                state['vein_strength'] = torch.ones_like(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            growth_rate = group['growth_rate']
            decay = group['decay']
            min_str = group['min_strength']
            max_str = group['max_strength']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                current_grad = p.grad.data
                prev_grad = state['previous_grad']
                vein_str = state['vein_strength']

                # Normalize gradients for directional comparison
                current_norm = torch.norm(current_grad)
                prev_norm = torch.norm(prev_grad)

                if current_norm > 0 and prev_norm > 0:
                    cos_sim = torch.dot(current_grad.flatten(), prev_grad.flatten()) / (current_norm * prev_norm)
                else:
                    cos_sim = torch.tensor(0.0)

                # Update vein strength based on directional consistency
                vein_str.mul_(1 + growth_rate * cos_sim)
                vein_str.mul_(decay)
                vein_str.clamp_(min=min_str, max=max_str)

                # Update parameters using vein-amplified gradients
                p.data.sub_(lr * vein_str * current_grad)

                # Store current gradient for next comparison
                state['previous_grad'].copy_(current_grad)
        return loss