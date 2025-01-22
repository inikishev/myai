# pylint:disable=signature-differs, not-callable
import torch
from torch.optim import Optimizer

class FractalOptimizer(Optimizer):
    """SGD but seems to waste 2 additional evaluates per step which doesn't help at all"""
    def __init__(self, params, lr=1e-3, fractal_lr=1e-4, beta=1.0, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if fractal_lr < 0.0:
            raise ValueError(f"Invalid fractal learning rate: {fractal_lr}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta value: {beta}")

        defaults = dict(lr=lr, fractal_lr=fractal_lr, beta=beta, eps=eps)
        super(FractalOptimizer, self).__init__(params, defaults)

    def generate_fractal_perturbation(self, total_params, beta, eps, device):
        """Generates 1D fractal noise using FFT-based method with numerical stability."""
        # Frequency spectrum
        freqs = torch.fft.fftfreq(total_params, d=1.0, device=device)
        # Avoid division by zero and extreme values by adding epsilon
        scaled_freqs = torch.abs(freqs) + eps
        # Compute scaling factor with beta exponent
        scale = scaled_freqs ** (beta / 2)

        # Generate complex white noise
        white_noise = torch.randn(total_params, 2, device=device)
        white_noise = torch.view_as_complex(white_noise)
        # Apply scaling
        scaled_noise = white_noise / scale
        # Explicitly zero out DC component to maintain zero mean
        scaled_noise[0] = 0.0

        # Inverse FFT to spatial domain
        noise = torch.fft.ifft(scaled_noise).real
        # Normalize to zero mean and unit variance
        if noise.numel() > 1 and noise.std() > 1e-8:
            noise = (noise - noise.mean()) / noise.std()
        return noise

    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step with fractal exploration."""
        if closure is None:
            raise ValueError("Closure required for FractalOptimizer")

        # Initial loss and gradient computation
        with torch.enable_grad(): loss = closure()
        # loss.backward()

        # Collect all parameters
        params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
        if not params:
            return loss

        with torch.no_grad():
            original_params = [p.detach().clone() for p in params]
            total_elements = sum(p.numel() for p in params)
            device = params[0].device

            # Generate stabilized fractal perturbation
            group = self.param_groups[0]
            perturbation = self.generate_fractal_perturbation(
                total_elements, group['beta'], group['eps'], device
            )
            perturbation *= group['fractal_lr']

            # Split perturbation into parameter components
            perts, index = [], 0
            for p in params:
                num = p.numel()
                perts.append(perturbation[index:index+num].view_as(p))
                index += num

            # Evaluate perturbation directions
            for p, pert in zip(params, perts):
                p.add_(pert)
            loss_plus = closure(False)

            for p, orig in zip(params, original_params):
                p.copy_(orig)
            for p, pert in zip(params, perts):
                p.sub_(pert)
            loss_minus = closure(False)

            for p, orig in zip(params, original_params):
                p.copy_(orig)

            # Determine optimal perturbation direction
            if loss_plus < loss and loss_plus < loss_minus:
                direction = 1
            elif loss_minus < loss:
                direction = -1
            else:
                direction = 0

            # Update parameters with gradient and fractal component
            pert_idx = 0
            for group in self.param_groups:
                lr = group['lr']
                fractal_lr = group['fractal_lr']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    # Standard gradient descent
                    p.data.add_(-lr * p.grad.data)
                    # Add fractal exploration
                    if direction != 0 and pert_idx < len(perts):
                        p.data.add_(perts[pert_idx], alpha=fractal_lr * direction)
                        pert_idx += 1

            # Reset gradients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

        return loss