# pylint:disable=signature-differs, not-callable
"""DeepSeek"""
import itertools
import random
from collections.abc import Callable

import numpy as np
import torch
import torchzero as tz
from torch import nn


# region ReliableGradientOptimizer
class ReliableGradientOptimizer(torch.optim.Optimizer):
    """Performs gradient descent unless gradient norm or variance is too low/too high.
    Then it performs a parameter-wise random vector line search.

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
                grad_norm = torch.norm(grad)
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
                        else:
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
                        else:
                            # Revert to old parameters
                            p.data = p.data_old
                            step_size *= line_search_decrease_factor
                        if step_size < line_search_min_step_size:
                            break
        return loss
#endregion

#region HillClimbing
class HillClimbing(tz.core.TensorListOptimizer):
    """
    tests all straight directions. Includes adaptive step size.

    Args:
        params (_type_): params
        lr (float, optional): initial step size. Defaults to 1.
        npos (float, optional): step size increase when good direction has been found. Defaults to 1.1.
        nneg (float, optional): step size decrease when no directions decreased loss. Defaults to 0.5.
    """
    def __init__(self, params, lr: float=1, npos = 1.2, nneg = 0.5):
        super().__init__(params, {})
        self.npos = npos
        self.nneg = nneg
        self.step_size = lr

    @torch.no_grad
    def step(self, closure):
        p = self.get_params().with_requires_grad()
        params = p.to_vec()

        def objective_fn(vec):
            p.from_vec_(vec)
            return closure(False)

        best_value = objective_fn(params)
        best_direction = None

        for i in range(params.numel()):
            for direction in [(i, 1), (i, -1)]:
                # Perturb parameters in the current direction
                perturbed_params = params.clone()
                perturbed_params[direction[0]] += direction[1] * self.step_size
                value = objective_fn(perturbed_params)

                if value < best_value:
                    best_value = value
                    best_direction = direction

        if best_direction is not None:
            # Update parameters in the best direction found
            params[best_direction[0]] += best_direction[1] * self.step_size
            p.from_vec_(params)
            self.step_size *= self.npos
        else:
            self.step_size *= self.nneg
            # No improvement found, stop optimization

        return best_value
#endregion

#region ExhaustiveHillClimbing
class ExhaustiveHillClimbing(tz.core.TensorListOptimizer):
    """Test all straight and all diagonal directions.

    Args:
        params (_type_): _description_
        lr (float, optional): initial step size. Defaults to 0.01.
        npos (float, optional): step size increase when good direction has been found. Defaults to 1.1.
        nneg (float, optional): step size decrease when no directions decreased loss. Defaults to 0.5.
        max_tree_size (_type_, optional):
            maximum directions to generate (i added this to avoid OOMs and freezing the device). Defaults to 1_000_000.
    """
    def __init__(self, params, lr=0.01, npos = 1.2, nneg = 0.5, max_tree_size=1_000_000, ):
        super().__init__(params, {})
        self.step_size = lr
        self.npos = npos
        self.nneg = nneg
        self.dim = self.get_params().total_numel()

        # Define behavior tree: includes individual, pairwise, and main diagonal perturbations
        self.max_tree_size = max_tree_size
        self.behavior_tree = self.build_behavior_tree()

    def build_behavior_tree(self):
        """
        Build an enhanced behavior tree with individual, pairwise, and main diagonal perturbations.
        """
        tree = []
        n_dels = 0

        # Individual parameter perturbations
        for i in range(self.dim):
            tree.append(((i,), 1))   # Perturb parameter i by +step_size
            tree.append(((i,), -1))  # Perturb parameter i by -step_size
            if len(tree) > self.max_tree_size:
                del tree[random.randrange(0, len(tree))]
                n_dels += 1
                if n_dels == self.max_tree_size: break

        n_dels = 0
        # Pairwise parameter perturbations
        for pair in itertools.combinations(range(self.dim), 2):
            tree.append((pair, 1))   # Perturb parameters in pair by +step_size
            tree.append((pair, -1))  # Perturb parameters in pair by -step_size
            if len(tree) > self.max_tree_size:
                del tree[random.randrange(0, len(tree))]
                n_dels += 1
                if n_dels == self.max_tree_size: break

        # Main diagonal perturbations
        tree.append((tuple(range(self.dim)), 1))   # Perturb all parameters by +step_size
        tree.append((tuple(range(self.dim)), -1))  # Perturb all parameters by -step_size

        return tree

    @torch.no_grad
    def step(self, closure):
        """
        Perform optimization using the zeroth order optimizer.

        :param objective_fn: Function to minimize, takes parameters and returns a scalar
        """
        # for iteration in range(self.max_iterations):
        p = self.get_params()
        params = p.to_vec()

        def objective_fn(vec):
            p.from_vec_(vec)
            return closure(False)

        best_value = objective_fn(params)
        best_direction = None

        for direction, sign in self.behavior_tree:
            # Perturb parameters in the current direction
            perturbed_params = params.clone()
            for idx in direction:
                perturbed_params[idx] += sign * self.step_size
            value = objective_fn(perturbed_params)

            if value < best_value:
                best_value = value
                best_direction = (direction, sign)

        if best_direction is not None:
            # Update parameters in the best direction found
            for idx in best_direction[0]:
                params[idx] += best_direction[1] * self.step_size
                p.from_vec_(params)

            self.step_size *= self.npos
            # print(f"Iteration {iteration+1}: New best value {best_value}")
        else:
            # No improvement found, stop optimization
            # print(f"Iteration {iteration+1}: No improvement. Stopping.")
            self.step_size *= self.nneg

        return best_value
#endregion

# region RLOptimizer
# Define the neural network to suggest descent directions
class DirectionSuggester(nn.Module):
    def __init__(self, input_size):
        super(DirectionSuggester, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * input_size),  # Output mean and log_std
        )

    def forward(self, x):
        output = self.fc(x)
        mean, log_std = output.chunk(2, dim=-1)
        return mean, log_std


# Zeroth order optimizer class
class RLOptimizer(tz.core.TensorListOptimizer):
    """A little RL model optimizes the function. Needs a lot of model_opt_cls tuning!

    Args:
        params (_type_): _description_
        lr (float, optional): learning rate. Defaults to 0.01.
        model (torch.nn.Module | None, optional):
            should accept num_variables sized vector, and return a 2 times longer vector.
            If none defaults to a simple linear model (but a better model is recommended). Defaults to None.
        model_opt_cls (Callable): RL model optimizer class. Defauts to `lambda p: torch.optim.AdamW(p, 1e-3)`
    """

    def __init__(
        self,
        params,
        lr=0.01,
        model: torch.nn.Module | None = None,
        model_opt_cls: Callable = lambda p: torch.optim.AdamW(p, 1e-3),
    ):
        super().__init__(params, {})
        if model is None: model = DirectionSuggester(self.get_params().total_numel())
        self.model = model.to(self.get_params()[0])
        self.lr = lr
        #self.model_optimizer = optim.Adam(model.parameters(), lr=model_lr)
        self.model_optimizer = model_opt_cls(model.parameters())

    @torch.no_grad
    def step(self, closure): # pylint:disable=signature-differs
        # Get current parameters
        p = self.get_params()
        current_params = p.to_vec()

        self.new_params = None

        with torch.enable_grad():
            self._f_after = float('inf')
            def model_closure(backward=True):
                # Get suggested mean and log_std from the model
                mean, log_std = self.model(current_params)

                # Sample direction d from N(mean, exp(log_std)^2)
                std = torch.exp(log_std)
                std = std.abs().nan_to_num(0,0,0)
                normal = torch.distributions.Normal(mean, std)
                d = normal.sample()

                # Compute log probability of d
                log_prob = normal.log_prob(d).sum(-1, keepdim=True)

                # Update parameters with suggested direction
                self.new_params = current_params - self.lr * d.squeeze()

                with torch.no_grad():
                    # Compute function values before and after update
                    f_before = closure(False)
                    p.from_vec_(self.new_params)
                    self._f_after = closure(False)

                # Compute improvement
                delta = f_before - self._f_after

                # Compute loss as -delta * log_prob
                loss = -delta * log_prob

                # Zero gradients and backpropagate
                if backward:
                    self.model_optimizer.zero_grad()
                    loss.backward()

                with torch.no_grad():
                    p.from_vec_(current_params)

                return loss

            self.model_optimizer.step(model_closure)

        assert self.new_params is not None
        p.from_vec_(self.new_params)
        return self._f_after
# endregion

# region SubspaceOptimizer
class SubspaceOptimizer(tz.core.TensorListOptimizer):
    """optimizes in a dynamic subspace (I don't know what kind of subspace)

    Args:
        params (_type_): _description_
        lr (_type_, optional): initial step size. Defaults to 1e-2.
        initial_subspace_dim (int, optional): initial dimension for subspace. Defaults to 10.
        step_size_factor (float, optional):
            step size multiplied or divided by this on successful/unsuccessful steps. Defaults to 1.5.
    """
    def __init__(self, params, lr=1e-2, step_size_factor=1.5, initial_subspace_dim=10, ):
        super().__init__(params, {})
        self.dim = self.get_params().total_numel()
        self.initial_subspace_dim = min(initial_subspace_dim, self.dim)
        self.step_size = lr
        self.step_size_factor = step_size_factor
        self._ref = self.get_params()[0]
        self.basis = self._initialize_basis()
        self.current_subspace_dim = self.initial_subspace_dim

    def _initialize_basis(self):
        # Initialize random orthogonal basis for the subspace
        basis = torch.randn(self.dim, self.initial_subspace_dim, device=self._ref.device, dtype=self._ref.dtype)
        q, _ = torch.linalg.qr(basis)
        return q

    def _expand_subspace(self):
        # Add a new random orthogonal basis vector
        new_vector = torch.randn(self.dim, device=self._ref.device, dtype=self._ref.dtype).unsqueeze(1)  # Make it a column vector
        projection = self.basis @ (self.basis.T @ new_vector)
        new_vector -= projection
        new_vector = new_vector.squeeze()  # Convert back to 1D tensor
        new_vector /= torch.linalg.vector_norm(new_vector)
        self.basis = torch.cat([self.basis, new_vector.unsqueeze(1)], dim=1)
        self.current_subspace_dim += 1
        # print("Expanded subspace to dimension:", self.current_subspace_dim)

    def _contract_subspace(self):
        # Remove a basis vector that contributes least to the subspace
        # For simplicity, remove the last added vector
        if self.current_subspace_dim > self.initial_subspace_dim:
            self.basis = self.basis[:, :-1]
            self.current_subspace_dim -= 1
            # print("Contracted subspace to dimension:", self.current_subspace_dim)

    @torch.no_grad
    def step(self, closure):
        p = self.get_params()
        current_params = p.to_vec()

        def objective_fn(vec):
            p.from_vec_(vec)
            return closure(False)

        current_value = objective_fn(current_params)
        best_value = float('inf')

        # Generate perturbation directions within the current subspace
        directions = [self.basis[:, i] for i in range(self.current_subspace_dim)]
        best_direction = None
        best_value = current_value

        for d in directions:
            perturbed_params = current_params + d * self.step_size
            value = objective_fn(perturbed_params)
            if isinstance(value, torch.Tensor): value = value.detach().cpu()
            if np.isfinite(value) and value < best_value:
                best_value = value
                best_direction = d

        if best_direction is not None:
            # Update parameters in the best direction
            current_params += best_direction * self.step_size
            self.step_size *= self.step_size_factor  # Increase step size on success
            # Optionally contract the subspace
            # if self.current_subspace_dim > self.initial_subspace_dim:
            self._contract_subspace()
            # print(f"Iteration {iteration}: New best value {best_value}")
        else:
            # No improvement in current subspace, expand subspace
            self._expand_subspace()
            self.step_size /= self.step_size_factor  # Decrease step size on failure

        p.from_vec_(current_params)
        return best_value
# endregion

#region Powell
class Powell(tz.core.TensorListOptimizer):
    """
    Powell's method.

    Parameters:
        params ( ): The parameters to optimize
        lr (float): Initial step size for line search, gets adapted based on successful/unsuccessful steps.
    """
    def __init__(self, params, lr=1.):
        super().__init__(params, {})
        self.delta = lr
        # Initialize search directions as standard basis vectors

    @torch.no_grad
    def step(self, closure):
        p = self.get_params().with_requires_grad()
        params = p.to_vec()

        n = params.numel()

        def func(vec):
            p.from_vec_(vec)
            return closure(False)

        f_old = func(params)
        f_new = None
        improvement = 0.0
        # Cycle through each search direction
        for i in range(n):
            d = torch.zeros_like(params)
            d[i] = 1

            # Evaluate function at current params and params + delta*d
            params_plus = params + self.delta * d
            f_plus = func(params_plus)
            # Determine the sign of improvement
            if f_plus < f_old:
                # Move in the positive direction
                while True:
                    params_new = params + self.delta * d
                    f_new = func(params_new)
                    if f_new < f_old:
                        params = params_new
                        f_old = f_new
                        self.delta *= 2  # Increase step size
                    else:
                        self.delta /= 2  # Decrease step size
                        break
                improvement += f_old - f_new
            else:
                params_minus = params - self.delta * d
                f_minus = func(params_minus)
                if f_minus < f_old:
                    # Move in the negative direction
                    while True:
                        params_new = params - self.delta * d
                        f_new = func(params_new)
                        if f_new < f_old:
                            params = params_new
                            f_old = f_new
                            self.delta *= 2  # Increase step size
                        else:
                            self.delta /= 2  # Decrease step size
                            break
                    improvement += f_old - f_new
                else:
                    # No improvement in this direction
                    self.delta /= 2

        p.from_vec_(params)
        return f_new if f_new is not None else f_old

# region RaySearch
def _raysearch_linesearch(closure, params: tz.TensorList, pvec, update, lr, loss, max_ls_iter, max_ars_iter):
    pvec.sub_(update, alpha=lr)
    params.from_vec_(pvec)
    loss2 = closure(False)
    if loss2 < loss:
        return lr * 2

    niter = 0
    init_lr = lr
    cur_lr = lr

    loss2 = closure(False)
    while loss2 > loss:
        lr /= 1.5
        cur_lr /= 2
        pvec.add_(update, alpha = cur_lr)
        niter += 1
        params.from_vec_(pvec)
        loss2 = closure(False)

        if niter == max_ls_iter:
            pvec.add_(update, alpha = cur_lr + init_lr)
            cur_lr = init_lr
            params.from_vec_(pvec)
            loss2 = closure(False)
            niter = 0

            if loss2 > loss:
                while loss2 > loss:
                    cur_lr /= 2
                    pvec.sub_(update, alpha = cur_lr)
                    params.from_vec_(pvec)
                    loss2 = closure(False)
                    niter += 1

                    if niter >= max_ls_iter:
                        pvec.sub_(update, alpha = cur_lr)
                        lr = init_lr
                        cur_lr = init_lr

                        update = params.grad.to_vec()
                        std = update.std()
                        if 0 < std < 1:
                            update /= std + 1e-6

                        pvec.sub_(update, alpha = cur_lr)
                        params.from_vec_(pvec)
                        loss2 = closure(False)

                        niter = 0
                        while loss2 > loss:
                            cur_lr /= 2
                            pvec.add_(update, alpha = cur_lr)
                            params.from_vec_(pvec)
                            loss2 = closure(False)

                            niter += 1
                            if niter >= max_ls_iter:
                                pvec.add_(update, alpha = cur_lr)

                                cur_lr = init_lr * 2
                                for i in range(max_ars_iter):
                                    vec = params.sample_like(cur_lr)

                                    params.add_(vec)
                                    if loss2 < loss: break
                                    params.sub_(vec)
                                    cur_lr /= 2
                                else:
                                    lr = init_lr * 2
                                    vec = params.sample_like(params.mean().abs_().clamp_(1e-3).mul_(0.1))
                                    params.add_(vec)
                                break
                        break
                # opposite direction decreased loss after line search
                else: lr = init_lr / 1.5
            # opposite direction instantly decreased loss
            else:
                lr = init_lr * 2
            break

    return lr


class RaySubspace(tz.core.TensorListOptimizer):
    """estimates a newton step in a meaningful subspace which spans gradient, momentum, difference between those, etc.

    Args:
        params (_type_): _description_
        lr (_type_, optional): lr. Defaults to 1e-3.
        num_directions (int, optional): _description_. Defaults to 3.
        epsilon (_type_, optional): _description_. Defaults to 1e-5.
        momentum (float, optional): _description_. Defaults to 0.9.
        paramwise (bool, optional): _description_. Defaults to False.
    """
    def __init__(self, params, lr=1e-3, num_directions=10, epsilon=1e-5, momentum=0.9, max_ls_iter = 4, max_ars_iter = 20, paramwise=False):
        defaults = dict(lr=lr, num_directions=num_directions, epsilon=epsilon, momentum=momentum)
        super().__init__(params, defaults)

        self.max_ls_iter = max_ls_iter

        self.init_lr = lr
        self.lrs = []
        self.max_ars_iter = max_ars_iter

        self.prev_grad = None; self.velocity = None; self.prev_update = None
        self.paramwise = paramwise

    @torch.no_grad()
    def _global_step(self, closure):
        lr = self.defaults['lr']
        num_directions = self.get_first_group_key('num_directions')
        epsilon = self.get_first_group_key('epsilon')
        momentum = self.get_first_group_key('momentum')


        with torch.enable_grad(): loss = closure()
        params = self.get_params().with_grad()
        p = params.to_vec()
        grad = params.grad.to_vec()
        # Initialize state for this parameter

        if self.prev_grad is None: self.prev_grad = torch.zeros_like(grad)
        if self.velocity is None: self.velocity = torch.zeros_like(grad)

        # Compute velocity
        self.velocity += grad
        self.velocity *= momentum

        # Store previous gradient
        self.prev_grad.copy_(grad)

        # Select directions
        directions = self.select_directions(p, grad, self.prev_grad, self.velocity, self.prev_update, num_directions)

        # Estimate Hessian in the selected directions
        hessian_estimates = self.estimate_hessian_global(p, directions, epsilon, closure, params)

        # Compute update step
        update = self.compute_update(grad, directions, hessian_estimates)

        # Reshape update to match parameter shape and apply
        update = update.view(p.shape)

        lr = self.defaults['lr'] = _raysearch_linesearch(closure, params, p, update, lr, loss, self.max_ls_iter, self.max_ars_iter)
        self.prev_update = update

        self.lrs.append(lr)
        if len(self.lrs) > 10:
            del self.lrs[0]
            if np.max(self.lrs) < self.init_lr / 100:
                lr = self.defaults['lr'] = self.init_lr
                self.lrs[-1] = lr

        return loss

    @torch.no_grad()
    def _paramwise_step(self, closure):
        with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            num_directions = group['num_directions']
            epsilon = group['epsilon']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.flatten()

                # Initialize state for this parameter
                state = self.state[p]
                if 'velocity' not in state:
                    state['velocity'] = torch.zeros_like(grad)
                if 'prev_update' not in state:
                    state['prev_update'] = torch.zeros_like(grad)

                # Compute velocity
                state['velocity'] = momentum * state['velocity'] + grad

                # Store previous gradient
                prev_grad = state['prev_grad']
                state['prev_grad'] = grad.clone()

                # Select directions
                directions = self.select_directions(p, grad, prev_grad, state['velocity'], state['prev_update'], num_directions)

                # Estimate Hessian in the selected directions
                hessian_estimates = self.estimate_hessian_paramwise(p, directions, epsilon, closure)

                # Compute update step
                update = self.compute_update(grad, directions, hessian_estimates)

                # Reshape update to match parameter shape and apply
                update = update.view(p.shape)
                state['prev_update'] = update
                p.add_(update, alpha=-lr)
        return loss

    @torch.no_grad
    def step(self, closure):
        if self.paramwise: return self._paramwise_step(closure)
        return self._global_step(closure)

    @torch.no_grad
    def select_directions(self, param, grad, prev_grad, velocity, prev_update, num_directions):
        directions = []

        # 1. Gradient direction
        norm = torch.linalg.vector_norm(grad)
        if norm > 0:
            grad_norm = grad/norm
            directions.append(grad_norm)
        else: grad_norm = None

        # 2. Velocity direction
        norm = torch.linalg.vector_norm(velocity)
        if norm > 0:
            velocity_norm = velocity/norm
            directions.append(velocity_norm)
        else: velocity_norm = None

        # 3. Previous gradient direction
        norm = torch.linalg.vector_norm(prev_grad)
        if norm > 0:
            prev_grad_norm = prev_grad / norm
            directions.append(prev_grad_norm)
        else: prev_grad_norm = None

        # 4. Previous update direction
        if prev_update is not None:
            norm = torch.linalg.vector_norm(prev_update)
            if norm > 0:
                prev_update_norm = prev_update / norm
                directions.append(prev_update_norm)
            else:
                prev_update_norm = None
        else:
            prev_update_norm = None

        # 5. Param itself
        norm = torch.linalg.vector_norm(param)
        if norm > 0: directions.append(param / norm)

        # 6, 7, 8. differences
        if grad_norm is not None and prev_grad_norm is not None:
            grad_diff = grad_norm - prev_grad_norm
            norm = torch.linalg.vector_norm(grad_diff)
            if norm > 0: directions.append(grad_diff / norm)

        if grad_norm is not None and velocity_norm is not None:
            grad_velocity_diff = grad_norm - velocity_norm
            norm = torch.linalg.vector_norm(grad_velocity_diff)
            if norm > 0: directions.append(grad_velocity_diff / norm)

        if prev_grad_norm is not None and prev_update_norm is not None:
            grad_update_diff = prev_grad_norm - prev_update_norm
            norm = torch.linalg.vector_norm(grad_update_diff)
            if norm > 0: directions.append(grad_update_diff / norm)

        if num_directions < len(directions):
            directions = random.choices(directions, k = num_directions) # this is not efficient but for now it will be it

        # 9+. Additional orthogonal directions if needed
        while len(directions) < num_directions:
            rand_dir = torch.randn_like(grad)
            for d in directions:
                rand_dir -= torch.dot(rand_dir, d) * d
            rand_dir /= rand_dir.norm() if rand_dir.norm() > 1e-8 else 1.0
            directions.append(rand_dir)

        return directions

    @torch.no_grad
    def estimate_hessian_global(self, pvec:torch.Tensor, directions:list[torch.Tensor], epsilon, closure, params: tz.tl.TensorList):
        hessian_estimates = []
        original_p = pvec.detach().clone()
        shape = pvec.shape
        for v in directions:
            # Perturb parameter in direction v
            params.from_vec_((original_p.view(-1) + epsilon * v).view(shape))
            self.zero_grad()
            with torch.enable_grad(): closure()
            grad_plus = params.grad.to_vec()

            params.from_vec_((original_p.view(-1) - epsilon * v).view(shape))
            self.zero_grad()
            with torch.enable_grad(): closure()
            grad_minus = params.grad.to_vec()

            pvec.copy_(original_p)
            # Finite difference Hessian-vector product
            Hv = (grad_plus - grad_minus) / (2 * epsilon)
            hessian_estimates.append(Hv)
        return hessian_estimates

    @torch.no_grad
    def estimate_hessian_paramwise(self, p:torch.Tensor, directions:list[torch.Tensor], epsilon, closure):
        hessian_estimates = []
        original_p = p.detach().clone()
        shape = p.shape
        for v in directions:
            # Perturb parameter in direction v
            p.copy_(original_p.view(-1) + epsilon * v).view(shape)
            self.zero_grad()
            with torch.enable_grad(): closure()
            assert p.grad is not None
            grad_plus = p.grad.detach().flatten()

            p.copy_(original_p.view(-1) - epsilon * v).view(shape)
            self.zero_grad()
            with torch.enable_grad(): closure()
            grad_minus = p.grad.detach().flatten()

            p.copy_(original_p)
            # Finite difference Hessian-vector product
            Hv = (grad_plus - grad_minus) / (2 * epsilon)
            hessian_estimates.append(Hv)
        return hessian_estimates

    @torch.no_grad
    def compute_update(self, grad, directions, hessian_estimates):
        update = torch.zeros_like(grad)
        for v, Hv in zip(directions, hessian_estimates):
            denominator = torch.dot(Hv, v)
            if abs(denominator) > 1e-8:
                alpha = torch.dot(grad, v) / denominator
                update += alpha * v
        return update


# region bouncyball

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