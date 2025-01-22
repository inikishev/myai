# pylint:disable=signature-differs, not-callable

import copy
from collections import defaultdict

import numpy as np
import torch
from torch.optim import Optimizer


class EvolutionaryGradientOptimizer(Optimizer):
    """evolutionary algo with gradient information and no thought whatsoever put into it"""
    def __init__(self, params, lr=1e-3, population_size=5,
                 reset_interval=50, elite_ratio=0.4, noise_scale=0.1,
                 noise_decay=0.99):
        defaults = dict(lr=lr, population_size=population_size,
                        reset_interval=reset_interval, elite_ratio=elite_ratio,
                        noise_scale=noise_scale, noise_decay=noise_decay)
        super().__init__(params, defaults)

        self.population = []
        self.fitness = []
        self.step_counter = 0
        self.state = defaultdict(dict)

        # Initialize population
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['population'] = [p.detach().clone() for _ in range(group['population_size'])]
                param_state['grad_moments'] = torch.zeros_like(p.data)
                param_state['noise_scale'] = group['noise_scale']

    @torch.no_grad
    def step(self, closure):
        if closure is None:
            raise ValueError("Closure required for EvolutionaryGradientOptimizer")

        # Evaluate all population members
        losses = []
        for member_id in range(self.defaults['population_size']):
            # Set model to population member parameters
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        p.data.copy_(self.state[p]['population'][member_id])

            # Compute loss and gradients
            with torch.enable_grad(): loss = closure()
            # loss.backward()
            losses.append(loss.item())

            # Store gradients and update parameters
            for group in self.param_groups:
                lr = group['lr']
                for p in group['params']:
                    if p.grad is None:
                        continue

                    # Update population member with gradient descent
                    self.state[p]['population'][member_id].data.add_(p.grad.data, alpha = -lr)

                    # Update gradient moments for adaptive noise
                    self.state[p]['grad_moments'] = 0.9 * self.state[p]['grad_moments'] + 0.1 * p.grad.data.pow(2)

            self.zero_grad()

        # Evolutionary reset every N steps
        if self.step_counter % self.defaults['reset_interval'] == 0:
            self._evolutionary_reset(losses)

        # Update main parameters to best performer
        best_member = np.argmin(losses)
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    p.data.copy_(self.state[p]['population'][best_member])

        self.step_counter += 1
        return torch.tensor(min(losses))

    def _evolutionary_reset(self, losses):
        elite_num = int(self.defaults['elite_ratio'] * self.defaults['population_size'])
        elite_ids = np.argsort(losses)[:elite_num]

        for group in self.param_groups:
            noise_decay = group['noise_decay']
            for p in group['params']:
                population = self.state[p]['population']
                grad_variance = torch.sqrt(self.state[p]['grad_moments'] + 1e-8)

                # Reset non-elite members
                for member_id in range(len(population)):
                    if member_id in elite_ids:
                        continue

                    # Select random elite parent
                    parent = population[np.random.choice(elite_ids)]

                    # Create mutated offspring with adaptive noise
                    noise = (torch.randn_like(parent)
                             * self.state[p]['noise_scale']
                             * (1 + grad_variance))
                    population[member_id].data.copy_(parent.data + noise)

                # Decay noise scale
                self.state[p]['noise_scale'] *= noise_decay

    def zero_grad(self):
        """Prevent default zero_grad behavior - we handle gradients manually"""
        pass