# pylint:disable=not-callable
"""NotebookLM made this

Important Notes:
•
This is a high-level outline, not a complete, ready-to-use implementation.
•
You need to implement the cubic subproblem solver and line search according to your specific needs.
•
Consider adding options for different convergence criteria, such as a tolerance for the change in function values.
•
You will need to choose an appropriate adaptive scheme for the regularization parameter sigma.
•
For large-scale problems, using an approximation of the Hessian matrix might be more computationally feasible.
•
The example code lacks a full implementation of many parts.
•
You will need to define the objective function that returns the loss, gradient, and Hessian based on your specific problem.
This detailed breakdown should give you a solid starting point to implement Cubic Regularized Newton in PyTorch. Remember that the most appropriate implementation might require experimentation and tuning for your specific problem.
"""

import torch
from torch import optim
from torch.autograd import functional

from torchzero.utils.derivatives import (
    jacobian_and_hessian,
    jacobian_list_to_vec,
    hessian_list_to_mat,
)
from torchzero.core import TensorListOptimizer


def cubic_regularized_newton(
    objective_function,
    initial_params,
    sigma=1.0,
):
    params = initial_params.clone().requires_grad_(True)  # params must be a tensor
    # iteration = 0
    # while iteration < max_iterations:
    loss, grad, hessian = objective_function(params)
    # grad_norm = torch.norm(grad)
    # if grad_norm < tolerance:
    #     break

    # Implement Cubic subproblem solver
    def cubic_model_1d(r):
        return (
            torch.linalg.solve(
                hessian + sigma * r / 2 * torch.eye(hessian.shape[0]), grad
            )
        ).norm() - r

    # Simple bisection method for solving the 1D problem
    r_lower = 0
    r_upper = 100
    for i in range(20):
        r_mid = (r_lower + r_upper) / 2
        if cubic_model_1d(r_mid) > 0:
            r_lower = r_mid
        else:
            r_upper = r_mid

    r = (r_lower + r_upper) / 2
    step = -torch.linalg.solve(
        hessian + sigma * r / 2 * torch.eye(hessian.shape[0]), grad
    )

    # Line Search Implementation. This is a simple version.
    alpha = 1.0
    rho = 0.5
    c = 0.1

    # Backtracking line search
    loss_new = float('inf')
    cubic_model = float('inf')  # happy now, pylance?

    for i in range(20):
        with torch.no_grad():
            params_new = params + alpha * step
            loss_new, _, _ = objective_function(params_new)

            cubic_model = (
                loss
                + grad.dot(alpha * step)
                #+ 0.5 * alpha * step.T.dot(hessian.mv(alpha * step))
                + 0.5 * alpha * step.dot(hessian.mv(alpha * step))
                + sigma / 6 * (alpha * step).norm() ** 3
            )
            if loss_new <= cubic_model:
                params = params_new.requires_grad_(True)
                break
            alpha *= rho
    else:
        sigma *= 2

    # Adaptively adjust sigma (example):
    if loss_new >= cubic_model:
        sigma *= 2
    else:
        sigma = max(sigma / 2, 1e-8)  # Add a minimum


    return params, loss, sigma


class CubicRegularizedNewtonV2(TensorListOptimizer):
    def __init__(self, params, initial_sigma = 1.0):
        super().__init__(params, {})

        self.sigma = initial_sigma

    @torch.no_grad
    def step(self, closure):  # pylint:disable=signature-differs
        params = self.get_params()

        def func(x):
            params.from_vec_(x)
            with torch.enable_grad():
                loss = closure(False)
                Gl, Hl = jacobian_and_hessian([loss], params)
            G = jacobian_list_to_vec(Gl)
            H = hessian_list_to_mat(Hl)
            return loss, G, H

        x0 = params.to_vec()

        x, loss, self.sigma = cubic_regularized_newton(func, x0, self.sigma)
        params.from_vec_(x)

        return loss
