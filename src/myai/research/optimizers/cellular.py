# pylint:disable=signature-differs, not-callable

import torch
from torch.optim import Optimizer
import numpy as np

class CellularAutomaton(Optimizer):
    """normalizes gradients for each weight by neigbours USING A FOR LOOP enjoy"""
    def __init__(self, params, lr=0.01, neighborhood_size=1):
        defaults = dict(lr=lr, neighborhood_size=neighborhood_size)
        super(CellularAutomaton, self).__init__(params, defaults)

        # Flatten all parameters into a single vector for grid mapping
        self.params = [p for group in self.param_groups for p in group['params']]
        self.param_shapes = [p.shape for p in self.params]
        self.param_sizes = [p.numel() for p in self.params]

        # Total number of parameters
        self.total_size = sum(self.param_sizes)

        # Initialize parameter indices for each cell
        self.param_indices = []
        idx = 0
        for size in self.param_sizes:
            self.param_indices.append(range(idx, idx + size))
            idx += size

        # Neighborhood size defines how many neighbors to consider on each side
        self.neighborhood_size = neighborhood_size

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        # Collect gradients and parameters
        gradients = []
        params_flat = []
        for p in self.params:
            if p.grad is None:
                gradients.append(torch.zeros_like(p))
            else:
                gradients.append(p.grad.detach().clone().flatten())
            params_flat.append(p.detach().clone().flatten())

        # Concatenate into single vectors
        flat_gradients = torch.cat(gradients)
        flat_params = torch.cat(params_flat)

        # Update each parameter based on local neighborhood
        updated_params = flat_params.clone()
        for i in range(self.total_size):
            # Determine neighborhood indices
            left = max(0, i - self.neighborhood_size)
            right = min(self.total_size, i + self.neighborhood_size + 1)
            neighborhood_indices = range(left, right)

            # Compute the standard deviation of gradients in the neighborhood
            neighborhood_gradients = flat_gradients[neighborhood_indices]
            sigma_i = torch.std(neighborhood_gradients)

            # Avoid division by zero
            sigma_i = sigma_i if sigma_i > 0 else 1.0

            # Update the parameter
            updated_params[i] -= self.defaults['lr'] * flat_gradients[i] / sigma_i

        # Reshape the updated parameters back to their original shapes
        idx = 0
        for p, shape in zip(self.params, self.param_shapes):
            size = p.numel()
            p.data = updated_params[idx:idx+size].view(shape)
            idx += size

        return loss
