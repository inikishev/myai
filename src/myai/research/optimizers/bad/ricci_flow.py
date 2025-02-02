# pylint:disable=signature-differs, not-callable # type:ignore
import itertools

import torch
from torch.optim import Optimizer


class RicciFlowFD(Optimizer):
    """uses finite differences for hessian diagonal otherwise known is O(N) version with Hvps below"""
    def __init__(self, params, lr=1e-3, curvature_rate=0.1, delta=1e-4,
                 damping=1e-3, grad_clip=10.0):
        defaults = dict(lr=lr, curvature_rate=curvature_rate, delta=delta,
                        damping=damping, grad_clip=grad_clip)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['metric'] = torch.ones_like(p.data)
                state['avg_curvature'] = torch.zeros_like(p.data)  # For normalization
    @torch.no_grad
    def step(self, closure):

        with torch.enable_grad(): loss = closure() # Original gradients

        for group in self.param_groups:
            lr = group['lr']
            curvature_rate = group['curvature_rate']
            delta = group['delta']
            damping = group['damping']
            grad_clip = group['grad_clip']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                metric = state['metric']
                avg_curv = state['avg_curvature']
                original_data = p.data.clone()
                g = p.grad.data
                hessian_diag = torch.zeros_like(p.data)

                # --- Stabilization 1: Adaptive Delta ---
                # Use relative perturbation (1% of parameter value) with floor at `delta`
                relative_delta = torch.abs(original_data) * 0.01
                applied_delta = torch.where(relative_delta > delta, relative_delta, delta)

                # --- Compute Hessian with adaptive delta ---
                shape = p.data.shape
                indices = itertools.product(*[range(dim) for dim in shape])
                for idx in indices:
                    original = original_data[idx].item()
                    d = applied_delta[idx].item()

                    # Perturb upwards
                    p.data[idx] = original + d
                    loss_high = closure(False).item()

                    # Perturb downwards
                    p.data[idx] = original - d
                    loss_low = closure(False).item()

                    p.data[idx] = original  # Reset
                    hessian_diag[idx] = (loss_high - 2 * loss.item() + loss_low) / (d ** 2)

                # --- Stabilization 2: Curvature Normalization ---
                # Track exponential moving average of curvature magnitude
                avg_curv.mul_(0.9).add_(hessian_diag.abs(), alpha=0.1)
                normalized_hessian = hessian_diag / (avg_curv + 1e-16)  # Prevent division by zero

                # --- Stabilization 3: Damped Metric Update ---
                # Modified Ricci flow: ∂g/∂t = -Ric + damping*(1 - g)
                # This adds a restoring force to prevent metric collapse/explosion
                metric_update = metric * (1 - curvature_rate * normalized_hessian) + \
                              damping * (1 - metric)
                metric_update = torch.clamp(metric_update, min=0.1, max=10.0)  # Hard bounds
                state['metric'] = metric_update

                # --- Stabilization 4: Preconditioned Gradient Step ---
                # Clip gradients to prevent explosive updates
                g_clipped = torch.clamp(g, -grad_clip, grad_clip)
                # Update using inverse metric with safeguard
                p.data.addcdiv_(-lr, g_clipped, metric_update + 1e-8)

        return loss


class RicciFlow(Optimizer):
    def __init__(self, params, lr=1e-3, curvature_rate=0.1, hvp_samples=2,
                 damping=1e-2, grad_clip=5.0):
        defaults = dict(lr=lr, curvature_rate=curvature_rate,
                        hvp_samples=hvp_samples, damping=damping,
                        grad_clip=grad_clip)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['metric'] = torch.ones_like(p.data)
                state['avg_curvature'] = torch.zeros_like(p.data)
    @torch.no_grad
    def step(self, closure):
        with torch.enable_grad():
            loss = closure(False)

            # First backward pass (retain graph for HvP)
            loss.backward(retain_graph=True, create_graph=True)

            params = []
            grads = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        params.append(p)
                        grads.append(p.grad.clone())

            # Compute Hessian diagonal via Hutchinson estimator
            hessian_diags = [torch.zeros_like(p) for p in params]

            for _ in range(self.defaults['hvp_samples']):
                # Generate Rademacher vectors (-1/+1)
                v = [torch.randint_like(p, high=2) * 2 - 1 for p in params]

                # Compute HvP = ∇(g·v)
                gv = sum([(g * v_p).sum() for g, v_p in zip(grads, v)])
                Hv = torch.autograd.grad(gv, params, retain_graph=True)

                # Accumulate diagonal estimate: E[v⊙Hv] = diag(H)
                for i, (v_p, Hv_p) in enumerate(zip(v, Hv)):
                    hessian_diags[i] += (v_p * Hv_p).detach()

        # Average across samples
        for hd in hessian_diags:
            hd.div_(self.defaults['hvp_samples'])

        # Parameter updates
        for group in self.param_groups:
            for p in params:
                if p.grad is None:
                    continue

                state = self.state[p]
                metric = state['metric']
                avg_curv = state['avg_curvature']
                hessian_diag = hessian_diags[params.index(p)]

                # Curvature normalization
                avg_curv.mul_(0.9).add_(hessian_diag.abs(), alpha=0.1)
                normalized_hessian = hessian_diag / (avg_curv + 1e-16)

                # Damped metric update
                metric_update = metric * (1 - group['curvature_rate'] * normalized_hessian)
                metric_update.add_(group['damping'] * (1 - metric))
                metric_update.clamp_(0.1, 10.0)
                state['metric'] = metric_update

                # Preconditioned gradient step
                g = torch.clamp(p.grad, -group['grad_clip'], group['grad_clip'])
                p.data.addcdiv_(-group['lr'], g, metric_update + 1e-8)

        # Cleanup graph to prevent memory leaks
        for p in params:
            p.grad = None
        torch.cuda.empty_cache()

        return loss