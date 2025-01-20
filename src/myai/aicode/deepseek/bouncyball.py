# pylint:disable=signature-differs, not-callable
"""DeepSeek"""
import itertools
import random
from collections.abc import Callable

import numpy as np
import torch
import torchzero as tz
from torch import nn


def bouncy_ball_simulation(
    objective_func,
    position,
    velocity,
    gravity,
    delta=0.01,
    bounciness=0.8,
    air_resistance=0.01,
    dampening=0.99,
    epsilon=1e-6,
):
    """
    Simulates a bouncy ball rolling down a function landscape.

    Parameters:
    - objective_func: Function that takes a position tensor x (size n) and returns (y, df/dx).
    - initial_position: Initial position tensor of size n+1 (x and y).
    - delta: Time step size (default: 0.01).
    - g: Gravitational acceleration scalar (default: 9.81).
    - bounciness: Bounciness coefficient (default: 0.8).
    - air_resistance: Air resistance factor (default: 0.01).
    - dampening: Dampening factor (default: 0.99).
    - num_steps: Total number of simulation steps (default: 1000).
    - epsilon: Tolerance for binary search convergence (default: 1e-6).

    Returns:
    - trajectory: List of position tensors over time.
    """
    n = position.size(0) - 1

    # for _ in range(num_steps):
    # Compute acceleration
    acc_gravity = gravity
    acc_air = -air_resistance * velocity
    acceleration = acc_gravity + acc_air

    # Predict next position and velocity
    position_next = position + velocity * delta + 0.5 * acceleration * delta**2
    velocity_next = velocity + acceleration * delta

    # Check for collision
    x_next = position_next[:n]
    y_next = position_next[n]
    f_x_next, df_dx_next = objective_func(x_next)

    if y_next < f_x_next:
        # Collision occurred within the step, find exact collision time tau
        tau_low = 0.0
        tau_high = delta
        for _ in range(20):  # Perform binary search iterations
            tau_mid = (tau_low + tau_high) / 2
            position_mid = position + velocity * tau_mid + 0.5 * acceleration * tau_mid**2
            x_mid = position_mid[:n]
            y_mid = position_mid[n]
            f_x_mid, _ = objective_func(x_mid)
            if y_mid < f_x_mid:
                tau_high = tau_mid
            else:
                tau_low = tau_mid
            if tau_high - tau_low < epsilon:
                break

        # Compute position and velocity at tau
        tau = (tau_low + tau_high) / 2
        position_tau = position + velocity * tau + 0.5 * acceleration * tau**2
        velocity_tau = velocity + acceleration * tau

        # Compute normal vector at collision point
        f_x_tau, df_dx_tau = objective_func(position_tau[:n])
        normal = torch.cat((df_dx_tau, torch.tensor([-1.0], dtype=df_dx_tau.dtype, device=df_dx_tau.device)))
        normal = normal / normal.norm()

        # Reflect velocity
        velocity_after = velocity_tau - (1 + bounciness) * (velocity_tau @ normal) * normal
        velocity_after = velocity_after * dampening  # Apply dampening

        # Simulate remaining time
        remaining_time = delta - tau
        if remaining_time > 0:
            acc_new = gravity + (-air_resistance * velocity_after)
            position = position_tau + velocity_after * remaining_time + 0.5 * acc_new * remaining_time**2
            velocity = velocity_after + acc_new * remaining_time
        else:
            position = position_tau
            velocity = velocity_after
    else:
        # No collision, update directly
        position = position_next
        velocity = velocity_next

    # Apply dampening to velocity (if needed)
    velocity = velocity * dampening

    return f_x_next, position, velocity

class BouncyBall(tz.core.TensorListOptimizer):
    def __init__(self, params, delta=0.01, g=9.81, bounciness=0.8, air_resistance=0.01, dampening=0.99, epsilon=1e-6):
        super().__init__(params, {})

        p = self.get_params()
        vec = p.to_vec()
        n = vec.numel()
        self.position = torch.cat((vec, torch.tensor([0.]).to(vec)))
        self.velocity = torch.zeros_like(self.position)
        self.gravity = torch.zeros(n+1).to(vec)
        self.gravity[-1] = -g  # Gravity acts downward in y
        self.trajectory = [self.position.detach().clone()]

        self.delta = delta
        self.g = g
        self.bounciness = bounciness
        self.air_resistence = air_resistance
        self.dampening = dampening
        self.epsilon = epsilon

    @torch.no_grad
    def step(self, closure):
        p = self.get_params()
        pvec = p.to_vec()

        def func(x):
            p.from_vec_(x)
            self.zero_grad()
            with torch.enable_grad(): loss = closure()
            grad = p.grad.to_vec()
            return loss, grad

        loss, self.position, self.velocity = bouncy_ball_simulation(
            func,
            position = self.position,
            velocity = self.velocity,
            gravity = self.gravity,
            delta = self.delta,
            bounciness = self.bounciness,
            air_resistance = self.air_resistence,
            dampening = self.dampening,
            epsilon = self.epsilon,
        )

        return loss
