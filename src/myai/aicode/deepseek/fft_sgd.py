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
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, filter_threshold=0.1):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        filter_threshold=filter_threshold)
        if momentum < 0 or dampening < 0 or lr < 0 or weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

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
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                # Apply FFT to the gradient
                d_p_fft = torch.fft.fftn(d_p)
                # Apply a low-pass filter in frequency domain
                # freq = torch.fft.fftfreq(d_p.size(0))
                mask = torch.ones_like(d_p_fft)
                for i in range(d_p.dim()):
                    freq_dim = torch.fft.fftfreq(d_p.size(i), d=1.0)
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
    def __init__(self, params, lr=1e-3, frequency_scaling=1.0):
        defaults = dict(lr=lr, frequency_scaling=frequency_scaling)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

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
                    freq_dim = torch.fft.fftfreq(d_p.size(i))
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