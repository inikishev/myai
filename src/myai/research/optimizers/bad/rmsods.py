# type:ignore # pylint:disable=signature-differs, not-callable

import math

import torch
from torch.optim import Optimizer


class RMSODS(Optimizer):
    """"Recursive Meta-Self-Optimizing Dynamical System" (RMSODS).

    this has no learning rate so if you forgot to remove second positional argument don't question why this is so unstable

    this is bad so if you use it don't question why it is so bad."""
    def __init__(self, params, sigma=5.0, rho=15.0, beta=1.5, epsilon=1e-8,
                 chaos_damping=0.3, max_energy=2.0):
        defaults = dict(sigma=sigma, rho=rho, beta=beta, epsilon=epsilon,
                        chaos_damping=chaos_damping, max_energy=max_energy)
        super().__init__(params, defaults)

        # Initialize chaotic system with constrained states
        self.state = []
        for group in self.param_groups:
            self.state.append([])
            for p in group['params']:
                state = {
                    'x': torch.randn_like(p) * 0.01,
                    'y': torch.randn_like(p) * 0.01,
                    'z': torch.randn_like(p) * 0.01,
                    'grad_momentum': torch.zeros_like(p),
                    'chaos_energy': torch.tensor(0.1),
                    'step': 0
                }
                self.state[-1].append(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group, gstate in zip(self.param_groups, self.state):
            sigma = group['sigma']
            rho = group['rho']
            beta = group['beta']
            epsilon = group['epsilon']
            damping = group['chaos_damping']
            max_energy = group['max_energy']

            for p, state in zip(group['params'], gstate):
                if p.grad is None:
                    continue

                grad = p.grad.data
                state['step'] += 1

                # Chaotic system stabilization mechanisms
                # 1. State normalization
                chaotic_states = torch.stack([state['x'], state['y'], state['z']])
                norms = torch.norm(chaotic_states, p=2, dim=0)
                chaotic_states = chaotic_states / (norms + epsilon).unsqueeze(0)
                state['x'], state['y'], state['z'] = chaotic_states

                # 2. Damped Lorenz equations
                dx = sigma*(state['y'] - state['x']) - damping*state['x']
                dy = state['x']*(rho - state['z']) - state['y'] - damping*state['y']
                dz = state['x']*state['y'] - beta*state['z'] - damping*state['z']

                # 3. Gradient-driven chaos control
                grad_correlation = torch.sum(
                    state['grad_momentum'] * grad / (torch.norm(state['grad_momentum'])*torch.norm(grad) + epsilon)
                )
                chaos_gain = torch.sigmoid(5 * grad_correlation)  # [0.5, 1.0] when correlated

                # Update chaotic states with controlled gain
                state['x'] += dx * 0.1 * chaos_gain
                state['y'] += dy * 0.1 * chaos_gain
                state['z'] += dz * 0.1 * chaos_gain

                # Energy calculation with constraints
                chaos_energy = (state['x']**2 + state['y']**2 + state['z']**2).sqrt()
                state['chaos_energy'] = 0.9*state['chaos_energy'] + 0.1*chaos_energy.clamp(max=max_energy)

                # Adaptive learning rate with warmup
                step = state['step']
                lr_base = 0.1 * (step / (step + 100))  # warmup
                lr = lr_base * torch.tanh(state['chaos_energy']) / (1 + math.log(step + 1))

                # Stabilized update with gradient momentum
                state['grad_momentum'] = 0.8*state['grad_momentum'] + 0.2*grad
                update = lr * state['grad_momentum'] / (state['chaos_energy'] + epsilon)

                # Apply update with gradient clipping
                update_norm = torch.norm(update)
                max_update = 0.1 * (1 + math.log(step + 1))
                if update_norm > max_update:
                    update = update * (max_update / (update_norm + epsilon))

                p.data.add_(-update)

        return loss