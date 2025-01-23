# type:ignore
import torch
from torch.optim import Optimizer

class StochasticLBFGS(Optimizer):
    """it is fairly stable and may beat lbfgs on some tasks.. but on minibatch tasks its just as bad.
    so is this really stochastic? Idk. Crucially this doesn't use closure."""
    def __init__(self, params, lr=1.0, warmup_steps=5, buffer_size=5,
                 min_damping=1e-3, max_lr_scale=10.0):
        defaults = dict(lr=lr, warmup_steps=warmup_steps, buffer_size=buffer_size,
                        min_damping=min_damping, max_lr_scale=max_lr_scale)
        super().__init__(params, defaults)

        self.state['step'] = 0
        self.state['buffer'] = []
        self.state['prev_params'] = None
        self.state['prev_grads'] = None

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Retrieve current parameters and gradients
        params = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)
                grads.append(p.grad.detach().clone())

        # Initialize previous state
        if self.state['prev_params'] is None:
            self.state['prev_params'] = [p.detach().clone() for p in params]
            self.state['prev_grads'] = [g.detach().clone() for g in grads]
            self.state['step'] += 1
            return loss

        # Compute s and y with 1-step delayed gradients
        s = [p - p_prev for p, p_prev in zip(params, self.state['prev_params'])]
        y = [g - g_prev for g, g_prev in zip(grads, self.state['prev_grads'])]

        # Apply adaptive damping
        sy = self._vdot(s, y)
        ss = self._vdot(s, s)
        damping = max(self.defaults['min_damping'], abs(sy)/(ss + 1e-8))
        y = [y_i + damping * s_i for y_i, s_i in zip(y, s)]

        # Store delayed (s, y) pair
        self.state['buffer'].append((s, y))
        if len(self.state['buffer']) > self.defaults['buffer_size']:
            self.state['buffer'].pop(0)

        # Save current state for next iteration (before parameter update)
        self.state['prev_params'] = [p.detach().clone() for p in params]
        self.state['prev_grads'] = [g.detach().clone() for g in grads]

        # Warmup phase: Use SGD with decaying learning rate
        if self.state['step'] < self.defaults['warmup_steps']:
            lr = self.defaults['lr'] * (self.state['step'] / self.defaults['warmup_steps'])
            self._sgd_update(params, grads, lr)
            self.state['step'] += 1
            return loss

        # L-BFGS phase
        d = self._two_loop_recursion(grads)
        self._adaptive_update(params, d, grads)

        self.state['step'] += 1
        return loss

    def _two_loop_recursion(self, grads):
        d = [-g.clone() for g in grads]
        alphas = []
        buffer = self.state['buffer']

        # First loop (reverse order)
        for i in reversed(range(len(buffer))):
            s_i, y_i = buffer[i]
            sy = self._vdot(s_i, y_i)
            rho = 1.0 / (sy + 1e-8)
            alpha = rho * self._vdot(s_i, d)
            alpha = torch.clamp(alpha, -1e3, 1e3)  # Clip extreme values
            alphas.append(alpha)
            d = [dj - alpha * yij for dj, yij in zip(d, y_i)]

        # Scale initial direction
        if len(buffer) > 0:
            s_last, y_last = buffer[-1]
            sy = self._vdot(s_last, y_last)
            yy = self._vdot(y_last, y_last)
            gamma = sy / (yy + 1e-8)
            gamma = torch.clamp(gamma, 1e-6, 1e6)  # Prevent division by near-zero
            d = [gamma * di for di in d]

        # Second loop (original order)
        for i in range(len(buffer)):
            s_i, y_i = buffer[i]
            sy = self._vdot(s_i, y_i)
            rho = 1.0 / (sy + 1e-8)
            beta = rho * self._vdot(y_i, d)
            alpha = alphas.pop()
            d = [dj + (alpha - beta) * sij for dj, sij in zip(d, s_i)]

        return d

    def _adaptive_update(self, params, direction, grads):
        # Auto-scale learning rate based on gradient magnitude
        grad_norm = sum(g.norm()**2 for g in grads).sqrt()
        dir_norm = sum(d.norm()**2 for d in direction).sqrt()
        lr_scale = torch.clamp(grad_norm / (dir_norm + 1e-8),
                              max=self.defaults['max_lr_scale'])
        lr = self.defaults['lr'] * lr_scale.item()

        # Apply update with NaN check
        for p, d in zip(params, direction):
            if torch.isnan(d).any():
                d = -p.grad  # Fallback to SGD on NaN
            p.add_(d, alpha=lr)

    def _sgd_update(self, params, grads, lr):
        for p, g in zip(params, grads):
            p.data.add_(g, alpha=-lr)

    def _vdot(self, a, b):
        return sum(torch.dot(ai.flatten(), bi.flatten()) for ai, bi in zip(a, b))