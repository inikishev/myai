# pylint:disable=not-callable
"""NotebookLM made this"""
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import functional
import numpy as np

from torchzero.utils.derivatives import (
    jacobian_and_hessian,
    jacobian_list_to_vec,
    hessian_list_to_mat,
)
from torchzero.core import TensorListOptimizer

class _CubicRegularizedNewton:
    """
    A class for implementing the Cubic Regularized Newton method.
    """

    def __init__(
        self,
        # tolerance=1e-5,
        # max_iterations=100,
        initial_sigma=1.0,
        # initial_hessian_sample_size=None,
        # hessian_sample_growth_rate=2,
        subproblem_solver_iterations=20,
        line_search_iterations=20,
    ):
        """
        Initializes the CubicRegularizedNewton optimizer.

        Args:
            tolerance (float): Tolerance for the gradient norm to achieve convergence.
            max_iterations (int): Maximum number of iterations.
            initial_sigma (float): Initial value for the regularization parameter.
             initial_hessian_sample_size (int): Initial number of samples when using sampled Hessian.
            hessian_sample_growth_rate (int): Growth rate of the Hessian sample size on each iteration.
            subproblem_solver_iterations (int): Number of iterations for the subproblem solver.
            line_search_iterations (int): Number of iterations for the line search.
        """
        # self.tolerance = tolerance
        # self.max_iterations = max_iterations
        self.sigma = initial_sigma
        # self.initial_hessian_sample_size = initial_hessian_sample_size
        # self.hessian_sample_growth_rate = hessian_sample_growth_rate
        self.subproblem_solver_iterations = subproblem_solver_iterations
        self.line_search_iterations = line_search_iterations

    def _cubic_subproblem_solver(self, hessian, grad, sigma):
        """
        Solves the cubic subproblem to find the step direction.

        This implementation uses a bisection method to find the root of the
        one-dimensional equation derived from the cubic model.

        Args:
          hessian (torch.Tensor): The Hessian matrix.
          grad (torch.Tensor): The gradient vector.
          sigma (float): The cubic regularization parameter.

        Returns:
            torch.Tensor: The computed step direction.
        """

        def cubic_model_1d(r):
            # The 1D equation we want to solve for r
            return (
                torch.linalg.solve(
                    hessian
                    + sigma * r / 2 * torch.eye(hessian.shape[0], device=hessian.device),
                    grad,
                )
            ).norm() - r

        # Bisection method for solving the 1D problem
        r_lower = 0
        r_upper = 100  # Initial upper bound for bisection
        for _ in range(self.subproblem_solver_iterations):
            r_mid = (r_lower + r_upper) / 2
            if cubic_model_1d(r_mid) > 0:
                r_lower = r_mid
            else:
                r_upper = r_mid

        r = (r_lower + r_upper) / 2
        step = -torch.linalg.solve(
            hessian + sigma * r / 2 * torch.eye(hessian.shape[0], device=hessian.device),
            grad,
        )
        return step

    def _backtracking_line_search(
        self, objective_function, params, step, sigma, loss, grad, hessian
    ):
        """
        Implements a backtracking line search to find an appropriate step size.

        Args:
            objective_function (callable): The objective function.
            params (torch.Tensor): Current parameters.
            step (torch.Tensor): Step direction.
            sigma (float): Current cubic regularization parameter.
            loss (float): Loss at current parameters.
            grad (torch.Tensor): Gradient at current parameters
            hessian (torch.Tensor): Hessian at current parameters

        Returns:
            tuple: (new parameters, new loss, success flag)
        """
        alpha = 1.0
        rho = 0.5
        c = 0.1

        for _ in range(self.line_search_iterations):
            with torch.no_grad():
                params_new = params + alpha * step
                loss_new, _, _ = objective_function(params_new)

                cubic_model = (
                    loss
                    + grad.dot(alpha * step)
                    + 0.5 * alpha * step.dot(hessian.mv(alpha * step))
                    + sigma / 6 * (alpha * step).norm() ** 3
                )
                if loss_new <= cubic_model:
                    return params_new.requires_grad_(True), loss_new, True
                alpha *= rho
        return params, loss, False

    def _approximate_hessian(self, objective_function, params, sample_size):
        """
        Approximates the Hessian using a sub-sampling strategy.

        This strategy computes the Hessian based on a random subset of the
        objective function's components.

        Args:
            objective_function (callable): The objective function which should return a tuple
            of loss, gradient and hessian (or a function which can be used to compute the hessian).
            params (torch.Tensor): Current parameters.
            sample_size (int): Number of samples for the hessian approximation.

        Returns:
            torch.Tensor: The approximated Hessian matrix.
        """
        if sample_size is None:
            _, _, hessian = objective_function(params)
            return hessian
        else:
            with torch.no_grad():
                full_loss, full_grad, _ = objective_function(
                    params
                )  # Get the full loss and gradient

            num_components = full_grad.shape
            if (
                sample_size >= num_components
            ):  # check if sample size is bigger than number of features
                _, _, full_hessian = objective_function(params)
                return full_hessian

            indices = np.random.choice(num_components, size=sample_size, replace=False)

            def sampled_objective(x):
                loss, grad, _ = objective_function(x)
                sampled_grad = grad[indices]

                # Construct a function that returns only the sampled gradient and is also suitable for hessian calculation:
                def sampled_grad_fn(x):
                    loss, grad, _ = objective_function(x)
                    return grad[indices]

                # Use torch.autograd.functional.hessian to compute the hessian of the sampled components
                sampled_hessian = functional.hessian(sampled_grad_fn, x)
                return loss, sampled_grad, sampled_hessian

            _, _, sampled_hessian = sampled_objective(params)  # compute sampled hessian
            # Scale up the sampled Hessian to represent an approximation of the full Hessian
            return sampled_hessian * (num_components / sample_size)

    def step(self, func, x: torch.Tensor):
        """
        Minimizes the objective function using the Cubic Regularized Newton method.

        Args:
            objective_function (callable): The objective function, should return a tuple
            of loss, gradient (torch.Tensor) and hessian (torch.Tensor)
            initial_params (torch.Tensor): Initial parameters for optimization.

        Returns:
            tuple: Optimized parameters, number of iterations, and the final loss value.
        """
        params = x.clone().requires_grad_(
            True
        )  # Ensure params are a tensor and track gradients
        # iteration = 0
        # hessian_sample_size = self.initial_hessian_sample_size

        loss, grad, hessian = func(
            params
        )  # Get loss and full gradient to check for convergence
        # grad_norm = torch.norm(grad)  # compute gradient norm

        # if grad_norm < self.tolerance:  # check convergence criterion
        #     break

        # Approximate the Hessian
        # hessian = self._approximate_hessian(
        #     objective_function, params, hessian_sample_size
        # )

        # Solve the cubic subproblem
        step = self._cubic_subproblem_solver(hessian, grad, self.sigma)

        # Perform backtracking line search
        params, loss, success = self._backtracking_line_search(
            func, params, step, self.sigma, loss, grad, hessian
        )

        # Adaptive adjustment of sigma
        if not success:
            self.sigma *= 2
        else:
            self.sigma = max(
                self.sigma / 2, 1e-8
            )  # Ensure sigma is non-zero and use a minimum

        # Adapt the hessian sample size for the next iteration
        # if self.initial_hessian_sample_size is not None:
        #     hessian_sample_size = min(
        #         int(hessian_sample_size * self.hessian_sample_growth_rate),
        #         len(grad),
        #     )


        return params.detach(), loss.item()


class CubicRegularizedNewtonV3(TensorListOptimizer):
    """3rd version of cubic regularized newton coded by NotebookLM after feeding it 10 papers about it.

    Args:
        params (params): params.
        initial_sigma (float): Initial value for the regularization parameter.
        subproblem_solver_iterations (int): Number of iterations for the subproblem solver.
        line_search_iterations (int): Number of iterations for the line search.
    """
    def __init__(
        self,
        params,
        initial_sigma=1.0,
        # initial_hessian_sample_size=None,
        # hessian_sample_growth_rate=2,
        subproblem_solver_iterations=20,
        line_search_iterations=20,
    ):
        super().__init__(params, {})

        self.solver = _CubicRegularizedNewton(
            initial_sigma=initial_sigma,
            subproblem_solver_iterations=subproblem_solver_iterations,
            line_search_iterations=line_search_iterations,
        )

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


        x, loss = self.solver.step(func, x0)
        params.from_vec_(x)

        return loss
