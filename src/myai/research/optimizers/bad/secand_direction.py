import torch
from torch.optim import Optimizer

class SecantDirectionOptimizer(Optimizer):
    """secant minimization in random direction why no use gradient direction because thats just line search
    well now its just line search in random direction but one that uses gradients."""
    def __init__(self, params, num_steps=3, init_step_size=0.1):
        defaults = dict(num_steps=num_steps, init_step_size=init_step_size)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step using random directional root-finding."""
        with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            params = group['params']
            num_steps = group['num_steps']
            alpha1 = group['init_step_size']

            # Capture original shapes and values
            with torch.no_grad():
                original_shapes = [p.data.shape for p in params]
                original_params = [p.data.clone() for p in params]
                flat_grads = torch.cat([p.grad.data.view(-1) for p in params])
                flat_params = torch.cat([p.data.view(-1) for p in params])

            # Generate random direction and normalize
            d = torch.randn_like(flat_params)
            d_norm = d.norm()
            if d_norm < 1e-8:
                d = torch.randn_like(flat_params)
            d_normalized = d / d.norm()

            # Function to compute directional derivative at x + αd
            def compute_g(alpha):
                # Perturb parameters
                perturbed = flat_params + alpha * d_normalized

                # Restore parameters to compute gradient
                offset = 0
                with torch.no_grad():
                    for p, shape, size in zip(params, original_shapes, [p.numel() for p in params]):
                        p.data.copy_(perturbed[offset:offset+size].view(shape))
                        offset += size

                # Compute new gradient
                with torch.enable_grad(): closure()
                flat_grad_perturbed = torch.cat([p.grad.data.view(-1) for p in params])

                # Reset parameters and gradients
                with torch.no_grad():
                    for p, orig in zip(params, original_params):
                        p.data.copy_(orig)
                    for p in params:
                        p.grad.detach_()
                        p.grad.zero_()

                return torch.dot(d_normalized, flat_grad_perturbed).item()

            # Secant method to find root of g(α)
            alpha0 = 0.0
            g0 = compute_g(alpha0)
            g1 = compute_g(alpha1)

            for _ in range(num_steps):
                if abs(g1 - g0) < 1e-6:
                    break
                denominator = g1 - g0
                if denominator == 0:
                    break
                alpha_next = alpha1 - g1 * (alpha1 - alpha0) / denominator
                alpha0, alpha1 = alpha1, alpha_next
                g0, g1 = g1, compute_g(alpha1)

            # Apply final update
            with torch.no_grad():
                final_update = alpha1 * d_normalized
                offset = 0
                for p, shape, size in zip(params, original_shapes, [p.numel() for p in params]):
                    p.data.add_(final_update[offset:offset+size].view(shape))
                    offset += size

        return loss