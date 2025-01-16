# pylint:disable=not-callable
"""NotebookLM (slightly edited)"""
import torch
from torchzero.utils.derivatives import jacobian_and_hessian, jacobian_list_to_vec, hessian_list_to_mat
from torchzero.core import TensorListOptimizer

def cubic_regularization_newton(func, x0, M, L0):
    """
    Applies the cubic regularization of Newton's method to minimize a function.

    Args:
        func (callable): A function that takes a PyTorch tensor as input and returns a scalar tensor
                        representing the function value, gradient and hessian.
        x0 (torch.Tensor): The starting point for the optimization.
        L0 (float): Lower bound for the Lipschitz constant of the Hessian.
        L (float):  Upper bound for the Lipschitz constant of the Hessian.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        torch.Tensor: The optimized value of x
    """
    x = x0.clone().detach().requires_grad_(True)
    # for k in range(max_iter):
    f, g, H = func(x) # function returns the value, gradient and hessian

    # Function to solve for r
    def solve_r(M_val,g,H):
        r = torch.tensor(1.0, requires_grad=False)
        for _ in range(100):
            r_new = torch.linalg.norm(torch.linalg.solve(H+ (M_val * r / 2) * torch.eye(H.shape[0]), g))
            if torch.abs(r_new - r) < 1e-6:
                break
            r = r_new

        return r

    r = solve_r(M,g,H)

    # Compute the cubic regularized newton step
    delta_x = -torch.linalg.solve(H + (M * r / 2) * torch.eye(H.shape[0]), g)
    x_new = x + delta_x

    # Check for function decrease
    f_new,_,_ = func(x_new)

    while f_new > f:
        M = 2 * M
        r = solve_r(M,g,H)
        delta_x = -torch.linalg.solve(H + (M * r / 2) * torch.eye(H.shape[0]), g)
        x_new = x + delta_x
        f_new,_,_ = func(x_new)


    #Update x and M
    x = x_new.clone().detach().requires_grad_(True)
    M = max(M / 2, L0)

    # Check for convergence
    # if torch.linalg.norm(delta_x) < tol:
    #     break

    #print(f"Iteration {k+1}, f(x) = {f.item():.4f}")
    return x, M


class CubicRegularizedNewton(TensorListOptimizer):
    """Cubic regularized newton written by NotebookLM by sending it the Cubic Regularization of Newton's Method paper. I had to fix small things and I don't know if it is correct but it works.

    Args:
        params (params): params
        L0 (float): Lower bound for the Lipschitz constant of the Hessian.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
    """
    def __init__(self, params, L0 = 0.1):
        super().__init__(params, {})

        self.L0 = L0
        self.M = L0

    @torch.no_grad
    def step(self, closure): # pylint:disable=signature-differs

        params = self.get_params()

        self._last_loss = None
        def func(x):
            params.from_vec_(x)
            with torch.enable_grad():
                self._last_loss = closure(False)
                Gl, Hl = jacobian_and_hessian([self._last_loss], params)
            G = jacobian_list_to_vec(Gl)
            H = hessian_list_to_mat(Hl)
            return self._last_loss, G, H

        x0 = params.to_vec()

        x, self.M = cubic_regularization_newton(func, x0 = x0, L0 = self.L0, M = self.M)
        params.from_vec_(x)

        return self._last_loss