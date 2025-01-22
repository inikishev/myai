# pylint:disable=signature-differs, not-callable
"""DeepSeek"""
import itertools
import random
from collections.abc import Callable

import numpy as np
import torch
import torchzero as tz
from torch import nn


class FFTSGD(torch.optim.Optimizer):
    """smoothes the gradient using fft, unlike laplacian smoothing sgd this doesnt flatten gradient and applies spatially.

    very slow and horrible convergence."""
    def __init__(self, params, lr=1e-3, momentum:float=0, dampening:float=0,
                 weight_decay:float=0, nesterov=False, filter_threshold=0.1):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        filter_threshold=filter_threshold)
        if momentum < 0 or dampening < 0 or lr < 0 or weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            lr = group['lr']
            filter_threshold = group['filter_threshold']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p.add_(p, alpha=weight_decay)

                # Apply FFT to the gradient
                d_p_fft = torch.fft.fftn(d_p)
                # Apply a low-pass filter in frequency domain
                # freq = torch.fft.fftfreq(d_p.size(0))
                mask = torch.ones_like(d_p_fft)
                for i in range(d_p.dim()):
                    freq_dim = torch.fft.fftfreq(d_p.size(i), d=1.0, device=d_p_fft.device)
                    for dim in list(range(i)) + list(range(i+1, d_p.dim())): freq_dim = freq_dim.unsqueeze(dim)
                    mask = mask * (freq_dim.abs() < filter_threshold).float()
                d_p_fft = d_p_fft * mask
                # Inverse FFT to get filtered gradient
                d_p_filtered = torch.fft.ifftn(d_p_fft).real

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p_filtered).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p_filtered, alpha=1 - dampening)
                    if nesterov:
                        d_p_filtered = d_p_filtered + momentum * buf
                    else:
                        d_p_filtered = buf

                p.data.add_(d_p_filtered, alpha=-lr)

        return loss

class FrequencyOptimizer(torch.optim.Optimizer):
    """scales low frequencies in the gradient. very slow, convergence not better than SGD"""
    def __init__(self, params, lr=1e-3, frequency_scaling=1.0):
        defaults = dict(lr=lr, frequency_scaling=frequency_scaling)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            frequency_scaling = group['frequency_scaling']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Apply FFT
                d_p_fft = torch.fft.fftn(d_p)

                # Manipulate frequency components
                # Example: scale low frequencies more
                freq_scaling = torch.ones_like(d_p_fft)
                for i in range(d_p.dim()):
                    freq_dim = torch.fft.fftfreq(d_p.size(i), device=d_p_fft.device)
                    for dim in list(range(i)) + list(range(i+1, d_p.dim())): freq_dim = freq_dim.unsqueeze(dim)
                    freq_scaling = freq_scaling * (1 / (1 + frequency_scaling * torch.abs(freq_dim)))

                d_p_fft_scaled = d_p_fft * freq_scaling

                # Ensure Hermitian symmetry for real output
                d_p_fft_scaled = (d_p_fft_scaled + d_p_fft_scaled.conj()) / 2

                # Apply inverse FFT
                d_p_modified = torch.fft.ifftn(d_p_fft_scaled).real

                # Update parameters
                p.data.add_(d_p_modified, alpha=-lr)

        return loss

class FFTMomentum(torch.optim.Optimizer):
    """fft momentum, way worse than heavy ball and nesterov momentum + more expensive"""
    def __init__(self, params, lr=0.01, history_length=10):
        defaults = dict(lr=lr, history_length=history_length)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                p.grad_history = []

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            history_length = group['history_length']
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad_history.append(p.grad.data.clone())
                if len(p.grad_history) > history_length:
                    p.grad_history.pop(0)
                if len(p.grad_history) == history_length:
                    grad_history = torch.stack(p.grad_history, dim=0)
                    original_shape = grad_history.shape[1:]
                    grad_flattened = grad_history.view(history_length, -1)
                    fft = torch.fft.fft(grad_flattened, dim=0)
                    cutoff = int(history_length * 0.3)
                    fft[cutoff:-cutoff] = 0
                    filtered_grad_flattened = torch.fft.ifft(fft, dim=0).real
                    filtered_grad = filtered_grad_flattened.view(history_length, *original_shape)
                    update_direction = filtered_grad.mean(dim=0)
                    p.data.add_(update_direction, alpha=-lr)
                else:
                    p.data.add_(p.grad.data, alpha=-lr)
        return loss