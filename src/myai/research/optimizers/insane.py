# pylint:disable=signature-differs, not-callable
import math

import torch
from torch.optim.optimizer import Optimizer


class INSANE(Optimizer):
    """
    Final Armageddon-proof version with:
    - Quantum gradient containment fields
    - Hysteresis-based step acceptance
    - Fractional order damping
    - Multi-stage explosion prevention
    """

    def __init__(self, params, lr=1e-3, betas=(0.95, 0.99, 0.999), eps=1e-6, order=3,
                 curvature_window=10, max_trust_ratio=5.0, trust_smooth=0.9,
                 grad_clip=10.0, step_reject_threshold=5.0, explosion_factor=1e3,
                 hysteresis_steps=5, chaos_control=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, order=order,
                        curvature_window=curvature_window,
                        max_trust_ratio=max_trust_ratio,
                        trust_smooth=trust_smooth,
                        grad_clip=grad_clip,
                        step_reject_threshold=step_reject_threshold,
                        explosion_factor=explosion_factor,
                        hysteresis_steps=hysteresis_steps,
                        chaos_control=chaos_control)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['prev_grad'] = torch.zeros_like(p)
                state['prev_param'] = p.detach().clone()
                state['prev_loss'] = torch.tensor(float('inf'))
                state['loss_history'] = torch.full((group['hysteresis_steps'],), float('inf'),
                                                  device=p.device)

                # Initialize buffers with quantum noise
                state['grad_buffer'] = torch.randn(
                    (group['curvature_window'],) + p.shape,
                    dtype=p.dtype,
                    device=p.device
                ) * group['eps']
                state['param_buffer'] = torch.randn_like(state['grad_buffer']) * group['eps']
                state['buffer_ptr'] = 0
                state['buffer_full'] = False

                # State initialization with chaos control
                state['curvatures'] = [
                    torch.zeros_like(p)
                    for _ in range(group['order']-1)
                ]
                state['momentums'] = [
                    torch.zeros_like(p)
                    for _ in range(group['order'])
                ]
                state['trust_ratio'] = torch.ones_like(p)
                state['quantum_containment'] = torch.ones_like(p)
                state['explosion_count'] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is None:
            raise ValueError("Requires closure for loss monitoring")

        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            order = group['order']
            window_size = group['curvature_window']
            max_tr = group['max_trust_ratio']
            eps = group['eps']
            trust_smooth = group['trust_smooth']
            grad_clip = group['grad_clip']
            reject_threshold = group['step_reject_threshold']
            explosion_factor = group['explosion_factor']
            hysteresis_steps = group['hysteresis_steps']
            chaos_control = group['chaos_control']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                state['step'] += 1

                # ========== QUANTUM CONTAINMENT FIELD ==========
                containment = state['quantum_containment']
                grad = grad * containment

                # ========== MULTI-STAGE GRADIENT CLIPPING ==========
                g_norm = grad.norm().clamp_min(eps)
                if g_norm > grad_clip:
                    grad.mul_(grad_clip / g_norm)
                    containment.mul_(0.9)

                # ========== BUFFER MANAGEMENT ==========
                ptr = state['buffer_ptr']
                state['grad_buffer'][ptr] = grad.detach()
                state['param_buffer'][ptr] = p.data.detach()
                state['buffer_ptr'] = (ptr + 1) % window_size
                state['buffer_full'] = state['buffer_full'] or (state['buffer_ptr'] == 0)

                # Skip curvature estimation until buffer fills
                if not state['buffer_full']:
                    p.data.add_(grad, alpha=-lr * containment.mean())
                    continue

                # ========== CHAOTIC CURVATURE ESTIMATION ==========
                G = state['grad_buffer']
                X = state['param_buffer']
                curvatures = state['curvatures']
                momentums = state['momentums']
                trust_ratio = state['trust_ratio']

                # Compute higher-order terms with chaos control
                for k in range(1, order):
                    sg_filter = self._savitzky_golay_coeffs(window_size, k+1, p.device, p.dtype)
                    sg_filter = sg_filter * (1 + chaos_control * torch.randn_like(sg_filter))

                    dkX = (X * sg_filter.view(-1, *([1]*p.dim()))).sum(0)
                    dkG = (G * sg_filter.view(-1, *([1]*p.dim()))).sum(0)

                    # Quantum-regulated curvature estimation
                    denom = dkX.abs().add(containment * eps)
                    curv_update = dkG / denom.clamp_min(eps)
                    curv_update = torch.tanh(curv_update / explosion_factor) * explosion_factor

                    curvatures[k-1].mul_(betas[1]).add_(curv_update, alpha=1 - betas[1])

                # ========== FRACTIONAL ORDER DAMPING ==========
                precond_grad = grad.clone()
                for k in range(1, order):
                    Hk = curvatures[k-1]
                    damping = (1 + Hk.abs().sqrt()).pow(1/(k+1))
                    precond_grad = precond_grad / damping.clamp_min(eps)
                    precond_grad = torch.tanh(precond_grad / grad_clip) * grad_clip

                # ========== HYSTERESIS-BASED MOMENTUM ==========
                lr_pows = [1.0] + [lr**k / math.gamma(k+1) for k in range(1, order+1)]
                for k in range(order):
                    beta = betas[min(k, len(betas)-1)]
                    step_update = precond_grad * lr_pows[k]
                    step_update = torch.clamp(step_update, -grad_clip, grad_clip)
                    momentums[k].mul_(beta).add_(step_update, alpha=1 - beta)

                # ========== UPDATE CONSTRUCTION ==========
                update = torch.zeros_like(p)
                for k in range(order):
                    update += momentums[k] / lr_pows[k] if lr_pows[k] > eps else momentums[k]

                # ========== LOSS-BASED TRUST ADAPTATION ==========
                current_loss = loss.detach()
                state['loss_history'] = torch.roll(state['loss_history'], shifts=-1)
                state['loss_history'][-1] = current_loss

                loss_ratio = current_loss / state['prev_loss'].clamp_min(eps)
                trust_update = torch.sigmoid(-torch.log(loss_ratio)) * 2
                trust_ratio.mul_(trust_smooth).add_(trust_update, alpha=1 - trust_smooth)
                trust_ratio.clamp_(eps, max_tr)

                # ========== MULTI-STAGE EXPLOSION PREVENTION ==========
                if current_loss > state['prev_loss'] * reject_threshold:
                    state['explosion_count'] += 1
                    containment.mul_(0.5 ** state['explosion_count'])
                    curvatures[:] = [c * 0.8 for c in curvatures]
                    momentums[:] = [m * 0.5 for m in momentums]
                    p.data.copy_(state['prev_param'])
                else:
                    state['explosion_count'] = max(0, state['explosion_count'] - 1)
                    p.data.sub_(update * trust_ratio * containment)
                    containment.mul_(0.99).add_(0.01)

                # ========== HYSTERESIS COMMIT ==========
                if (state['loss_history'][-hysteresis_steps:] < current_loss).all():
                    state['prev_grad'].copy_(grad)
                    state['prev_param'].copy_(p.data)
                    state['prev_loss'] = current_loss

        return loss

    def _savitzky_golay_coeffs(self, window_size, deriv_order, device, dtype):
        """Correctly handles even/odd window sizes for exact dimension matching"""
        half_window = (window_size - 1) // 2

        # Generate centered indices for any window size
        x = torch.arange(window_size, dtype=torch.float64, device=device) - half_window

        # Create Vandermonde matrix with polynomial basis
        poly_order = deriv_order + 2  # Increased order for stability
        exponents = torch.arange(poly_order, dtype=torch.float64, device=device)
        A = x[:, None] ** exponents  # Shape: [window_size, poly_order]

        try:
            # Regularized pseudo-inverse calculation
            ATA = A.T @ A
            reg = torch.eye(ATA.shape[0], dtype=ATA.dtype, device=device) * 1e-10
            A_pinv = torch.linalg.pinv(ATA + reg) @ A.T
        except RuntimeError:
            # Fallback to standard pseudo-inverse
            A_pinv = torch.linalg.pinv(A)

        # Get coefficients for the requested derivative order
        coeffs = A_pinv[deriv_order].to(dtype=dtype)

        # Ensure coefficients match window size exactly
        assert len(coeffs) == window_size, \
            f"Filter length {len(coeffs)} != window size {window_size}"

        return coeffs / coeffs.abs().sum().clamp_min(1e-10)