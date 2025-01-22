# pylint:disable=signature-differs, not-callable
"""DeepSeek"""
import itertools
import random
from collections.abc import Callable

import numpy as np
import torch
import torchzero as tz
from torch import nn


class ZOBasisSubspace(tz.core.TensorListOptimizer):
    """optimizes in a dynamic subspace (I don't know what kind of subspace)

    Args:
        params (_type_): _description_
        lr (_type_, optional): initial step size. Defaults to 1e-2.
        initial_subspace_dim (int, optional): initial dimension for subspace. Defaults to 10.
        step_size_factor (float, optional):
            step size multiplied or divided by this on successful/unsuccessful steps. Defaults to 1.5.
    """
    def __init__(self, params, lr=1e-2, step_size_factor=1.5, initial_subspace_dim=10, ):
        super().__init__(params, {})
        self.dim = self.get_params().total_numel()
        self.initial_subspace_dim = min(initial_subspace_dim, self.dim)
        self.step_size = lr
        self.step_size_factor = step_size_factor
        self._ref = self.get_params()[0]
        self.basis = self._initialize_basis()
        self.current_subspace_dim = self.initial_subspace_dim

    def _initialize_basis(self):
        # Initialize random orthogonal basis for the subspace
        basis = torch.randn(self.dim, self.initial_subspace_dim, device=self._ref.device, dtype=self._ref.dtype)
        q, _ = torch.linalg.qr(basis)
        return q

    def _expand_subspace(self):
        # Add a new random orthogonal basis vector
        new_vector = torch.randn(self.dim, device=self._ref.device, dtype=self._ref.dtype).unsqueeze(1)  # Make it a column vector
        projection = self.basis @ (self.basis.T @ new_vector)
        new_vector -= projection
        new_vector = new_vector.squeeze()  # Convert back to 1D tensor
        new_vector /= torch.linalg.vector_norm(new_vector)
        self.basis = torch.cat([self.basis, new_vector.unsqueeze(1)], dim=1)
        self.current_subspace_dim += 1
        # print("Expanded subspace to dimension:", self.current_subspace_dim)

    def _contract_subspace(self):
        # Remove a basis vector that contributes least to the subspace
        # For simplicity, remove the last added vector
        if self.current_subspace_dim > self.initial_subspace_dim:
            self.basis = self.basis[:, :-1]
            self.current_subspace_dim -= 1
            # print("Contracted subspace to dimension:", self.current_subspace_dim)

    @torch.no_grad
    def step(self, closure):
        p = self.get_params()
        current_params = p.to_vec()

        def objective_fn(vec):
            p.from_vec_(vec)
            return closure(False)

        current_value = objective_fn(current_params)
        best_value = float('inf')

        # Generate perturbation directions within the current subspace
        directions = [self.basis[:, i] for i in range(self.current_subspace_dim)]
        best_direction = None
        best_value = current_value

        for d in directions:
            perturbed_params = current_params + d * self.step_size
            value = objective_fn(perturbed_params)
            if isinstance(value, torch.Tensor): value = value.detach().cpu()
            if np.isfinite(value) and value < best_value:
                best_value = value
                best_direction = d

        if best_direction is not None:
            # Update parameters in the best direction
            current_params += best_direction * self.step_size
            self.step_size *= self.step_size_factor  # Increase step size on success
            # Optionally contract the subspace
            # if self.current_subspace_dim > self.initial_subspace_dim:
            self._contract_subspace()
            # print(f"Iteration {iteration}: New best value {best_value}")
        else:
            # No improvement in current subspace, expand subspace
            self._expand_subspace()
            self.step_size /= self.step_size_factor  # Decrease step size on failure

        p.from_vec_(current_params)
        return best_value
