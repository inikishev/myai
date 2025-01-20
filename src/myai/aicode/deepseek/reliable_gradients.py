# pylint:disable=signature-differs, not-callable
import torch

class ReliableGradientOptimizer(torch.optim.Optimizer):
    """Performs gradient descent unless gradient norm or variance is too low/too high.
    Then it performs a parameter-wise random vector line search.

    (note this is not a very good one...)

    Args:
        params (_type_): _description_
        lr (_type_, optional): _description_. Defaults to 1e-3.
        momentum (float, optional): _description_. Defaults to 0.9.
        threshold_low (_type_, optional): _description_. Defaults to 1e-6.
        threshold_high (_type_, optional): _description_. Defaults to 1e2.
        variance_threshold (_type_, optional): _description_. Defaults to 1e-3.
        line_search_max_iterations (int, optional): _description_. Defaults to 10.
        line_search_decrease_factor (float, optional): _description_. Defaults to 0.5.
        line_search_min_step_size (_type_, optional): _description_. Defaults to 1e-6.
        armijo_c (float, optional): _description_. Defaults to 0.0001.
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, threshold_low=1e-6, threshold_high=1e2, variance_threshold=1e-3,
                 line_search_max_iterations=10, line_search_decrease_factor=0.5, line_search_min_step_size=1e-6,
                 armijo_c=0.0001):
        defaults = dict(lr=lr, momentum=momentum, threshold_low=threshold_low,
                        threshold_high=threshold_high, variance_threshold=variance_threshold,
                        line_search_max_iterations=line_search_max_iterations,
                        line_search_decrease_factor=line_search_decrease_factor,
                        line_search_min_step_size=line_search_min_step_size,
                        armijo_c=armijo_c)
        super().__init__(params, defaults)

        for g in self.param_groups:
            for p in g['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p)

    @torch.no_grad
    def step(self, closure):
        with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                lr = group['lr']
                momentum = group['momentum']
                threshold_low = group['threshold_low']
                threshold_high = group['threshold_high']
                variance_threshold = group['variance_threshold']
                line_search_max_iterations = group['line_search_max_iterations']
                line_search_decrease_factor = group['line_search_decrease_factor']
                line_search_min_step_size = group['line_search_min_step_size']
                armijo_c = group['armijo_c']

                # Check gradient reliability
                grad_norm = torch.linalg.vector_norm(grad)
                grad_variance = torch.var(grad)
                if grad_norm < threshold_low or grad_norm > threshold_high or grad_variance > variance_threshold:
                    # Gradients are unreliable
                    # Choose alternative direction (e.g., random direction)
                    alternative_direction = torch.randn_like(p.data)
                    # Perform line search without Armijo condition
                    step_size = lr
                    for _ in range(line_search_max_iterations):
                        # Save current parameters
                        p.data_old = p.data.clone()
                        # Proposed update
                        p.add_(-step_size * alternative_direction)
                        # Evaluate loss
                        new_loss = closure(False)
                        if new_loss < loss:
                            break  # Accept the step
                        # Revert to old parameters
                        p.data = p.data_old
                        step_size *= line_search_decrease_factor
                        if step_size < line_search_min_step_size:
                            break
                else:
                    # Gradients are reliable
                    # Compute update direction with momentum
                    buf = self.state[p]['momentum_buffer']
                    buf.mul_(momentum).add_(-grad)
                    update_direction = buf
                    # Compute gradient dot product with update direction
                    gradient_dot_update = (grad * update_direction).sum()
                    # Perform backtracking line search with Armijo condition
                    step_size = lr
                    for _ in range(line_search_max_iterations):
                        # Save current parameters
                        p.data_old = p.data.clone()
                        # Proposed update
                        p.data.add_(step_size * update_direction)
                        # Evaluate loss
                        new_loss = closure(False)
                        # Check Armijo condition
                        if new_loss <= loss + armijo_c * step_size * gradient_dot_update:
                            break  # Accept the step
                        # Revert to old parameters
                        p.data = p.data_old
                        step_size *= line_search_decrease_factor
                        if step_size < line_search_min_step_size:
                            break
        return loss