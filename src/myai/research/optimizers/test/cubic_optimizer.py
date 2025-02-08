"""i was testing new reasoning gemini flash and it went ahead and produced this"""
import torch
from torch.optim import Optimizer

class CubicLineSearch(Optimizer):
    """
    Approximates the loss function locally using a third-order Taylor expansion along the negative gradient direction
    and finds a step size by solving the quadratic equation derived from setting the derivative of the cubic approximation to zero.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (not directly used, but kept for consistency with Optimizer base class).
                                It's recommended to set lr to 1.0 as the step size is computed dynamically.
        h (float, optional): step size for finite difference approximation of higher-order derivatives.
                              A small value like 1e-3 or 1e-4 is recommended.

    """

    def __init__(self, params, lr=1.0, h=1e-4):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if h <= 0.0:
            raise ValueError("Invalid finite difference step size: {}".format(h))
        defaults = dict(lr=lr, h=h)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                h = group['h']
                d = -grad  # Direction is negative gradient

                x_k = p.data.clone()
                g_k = grad.clone()

                x_kh = x_k + h * d
                x_k2h = x_k + 2 * h * d

                def get_grad_eval(param_data):
                    param_backup = p.data.clone()
                    p.data = param_data
                    if closure is not None:
                        with torch.enable_grad():
                            closure_loss = closure(False)
                            closure_loss.backward(retain_graph=True) # retain_graph=True to allow multiple gradient computations in one step
                            current_grad = p.grad.data.clone()
                            p.grad.zero_() # Clear gradient after use
                    else:
                        raise ValueError("Closure must be provided for CubicOptimizer.")
                    p.data = param_backup # Restore original parameters
                    return current_grad

                g1 = get_grad_eval(x_kh)
                g2 = get_grad_eval(x_k2h)


                gd_gk = torch.dot(g_k.view(-1), d.view(-1))
                gd_g1 = torch.dot(g1.view(-1), d.view(-1))
                gd_g2 = torch.dot(g2.view(-1), d.view(-1))

                A = 0.5 * (gd_g2 - 2 * gd_g1 + gd_gk) / (h * h)
                B = (gd_g1 - gd_gk) / h
                C = gd_gk

                alpha = 0.0
                if abs(A) > 1e-8: # Avoid division by zero, use a small threshold
                    discriminant = B*B - 4*A*C
                    if discriminant >= 0:
                        alpha_1 = (-B + torch.sqrt(discriminant)) / (2 * A)
                        alpha_2 = (-B - torch.sqrt(discriminant)) / (2 * A)
                        alpha = alpha_1 # Heuristic: choose one root, e.g., the first one. More sophisticated root selection could be implemented.
                    else:
                        alpha = -C / B if abs(B) > 1e-8 else 0.0 # Fallback to linear approximation if complex roots or B is zero
                elif abs(B) > 1e-8:
                    alpha = -C / B
                else:
                    alpha = 0.0 # Gradient is already close to zero

                if not torch.isfinite(alpha):
                    alpha = 0.0 # If alpha is NaN or Inf, fall back to zero step


                p.data.add_(alpha * d)


        return loss


import torch
from torch.optim import Optimizer

class DiagonalCubicOptimizer(Optimizer):
    """
    Implements a Diagonal Cubic Optimizer based on cubic approximation,
    extending the 1D line search to a diagonal step size matrix.

    It approximates the loss function locally using a third-order Taylor expansion
    and calculates a diagonal step size matrix by independently solving a cubic minimization
    problem for each parameter dimension using function and gradient evaluations.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (not directly used, but kept for consistency with Optimizer base class).
                                Recommended to set lr to 1.0 as step sizes are computed dynamically.
        h (float, optional): step size for finite difference approximation of higher-order derivatives.
                              A small value like 1e-3 or 1e-4 is recommended.

    """

    def __init__(self, params, lr=1.0, h=1e-4):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if h <= 0.0:
            raise ValueError("Invalid finite difference step size: {}".format(h))
        defaults = dict(lr=lr, h=h)
        super(DiagonalCubicOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                h = group['h']
                d = -grad  # Direction is negative gradient

                x_k = p.data.clone()
                g_k = grad.clone()

                step_size_diag = torch.zeros_like(p.data) # Initialize diagonal step sizes

                param_shape = p.data.shape
                param_size = p.data.numel()
                param_flat = p.data.view(-1)
                grad_flat = grad.view(-1)


                def get_grad_eval_flat(param_data_flat):
                    param_backup = p.data.clone()
                    p.data = param_data_flat.view(param_shape)
                    if closure is not None:
                        with torch.enable_grad():
                            closure_loss = closure(False)
                            closure_loss.backward(retain_graph=True)
                            current_grad = p.grad.data.clone()
                            p.grad.zero_()
                    else:
                        raise ValueError("Closure must be provided for DiagonalCubicOptimizer.")
                    p.data = param_backup
                    return current_grad.view(-1) # Return flattened gradient

                for i in range(param_size):
                    # Direction for each parameter dimension
                    direction_vec = torch.zeros_like(param_flat)
                    direction_vec[i] = 1.0
                    d_i = direction_vec

                    x_kh_flat = param_flat + h * d_i
                    x_k2h_flat = param_flat + 2 * h * d_i

                    g1_flat = get_grad_eval_flat(x_kh_flat)
                    g2_flat = get_grad_eval_flat(x_k2h_flat)

                    gk_dot_di = torch.dot(grad_flat, d_i)
                    g1_dot_di = torch.dot(g1_flat, d_i)
                    g2_dot_di = torch.dot(g2_flat, d_i)

                    A = 0.5 * (g2_dot_di - 2 * g1_dot_di + gk_dot_di) / (h * h)
                    B = (g1_dot_di - gk_dot_di) / h
                    C = gk_dot_di

                    alpha_i = 0.0
                    if abs(A) > 1e-8:
                        discriminant = B*B - 4*A*C
                        if discriminant >= 0:
                            alpha_1 = (-B + torch.sqrt(discriminant)) / (2 * A)
                            # alpha_2 = (-B - torch.sqrt(discriminant)) / (2 * A)
                            alpha_i = alpha_1 # Choose one root
                        else:
                            alpha_i = -C / B if abs(B) > 1e-8 else 0.0
                    elif abs(B) > 1e-8:
                        alpha_i = -C / B
                    else:
                        alpha_i = 0.0

                    if not torch.isfinite(alpha_i):
                        alpha_i = 0.0

                    step_size_diag.view(-1)[i] = alpha_i


                p.data.add_((step_size_diag * d)) # Element-wise multiplication for diagonal scaling


        return loss

