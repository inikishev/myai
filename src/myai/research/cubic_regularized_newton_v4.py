"""Made by deepseep after an INSANE amount of thinking"""
import torch

from torchzero.utils.derivatives import jacobian_and_hessian, jacobian_list_to_vec, hessian_list_to_mat


class CubicRegularizedNewton(torch.optim.Optimizer):
    def __init__(self, params, mu=1e-3, lr=1.0):
        """uses autograd.functional.hessian"""
        defaults = dict(mu=mu, lr=lr)
        super().__init__(params, defaults)

    @staticmethod
    def _compute_step(g, H, mu):
        s = torch.zeros_like(g)
        eps = 1e-8  # Small epsilon to prevent division by zero
        for _ in range(3):  # Perform 3 iterations
            norm_s = torch.norm(s)
            damping = (1 / (mu + eps)) * norm_s
            A = H + damping * torch.eye(H.size(0), device=g.device)
            s = -torch.linalg.solve(A, g) # pylint:disable=not-callable
        return s

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mu = group['mu']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                H = torch.autograd.functional.hessian(closure, p)
                s = self._compute_step(g, H, mu)
                p.data.add_(lr * s)
        return loss

