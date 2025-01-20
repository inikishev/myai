# pylint:disable=signature-differs, not-callable

import torch

class GradientExtrapolationSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, window_size=10, adjust_factor = 1.):
        defaults = dict(lr=lr, window_size=window_size, adjust_factor=adjust_factor)
        super().__init__(params, defaults)

        # Initialize gradient history and parameter shapes per parameter
        self.gradient_history = []
        self.param_shapes = []
        self.params: list[torch.Tensor] | None = None

    def step(self,closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if self.params is None:
            self.params = [p for group in self.param_groups for p in group['params']]
            for p in self.params:
                self.gradient_history.append({'grads': []})
                self.param_shapes.append(p.shape)

        # Collect current gradients, flatten them, and store in history
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # Flatten the gradient
            flat_grad = p.grad.detach().flatten()
            # Store the gradient
            self.gradient_history[i]['grads'].append(flat_grad)
            # Limit history to window_size
            if len(self.gradient_history[i]['grads']) > self.defaults['window_size']:
                self.gradient_history[i]['grads'].pop(0)

        # Perform step for each parameter
        for i, p in enumerate(self.params):
            assert p.grad is not None
            state = self.state[p]
            lr = self.defaults['lr']
            adjust_factor = self.defaults['adjust_factor']
            grads = self.gradient_history[i]['grads']
            if len(grads) >= self.defaults['window_size']:
                # Predict the next gradient
                predicted_grad_flat = self.predict_gradient(grads)
                # Reshape the predicted gradient back to original shape
                predicted_grad = predicted_grad_flat.view(self.param_shapes[i])

                # Adjust the current gradient with the predicted future gradient
                if adjust_factor == 1: adjusted_grad = predicted_grad
                else:
                    adjusted_grad = p.grad.lerp(predicted_grad, adjust_factor)

                # Update the parameter
                p.data -= lr * adjusted_grad
            else:
                # Not enough history, perform standard SGD update
                p.data -= lr * p.grad

        return loss

    def predict_gradient(self, grads):
        # Stack the gradients into a 2D tensor
        grads_stack = torch.stack(grads, dim=0)  # Shape: (n, d)

        # Time steps from 0 to n-1
        n = grads_stack.size(0)
        T = torch.arange(n, dtype=grads_stack.dtype, device=grads_stack.device).unsqueeze(1)  # Shape: (n, 1)

        # Design matrix A with time steps and ones
        A = torch.cat([T, torch.ones_like(T)], dim=1)  # Shape: (n, 2)

        # Solve A @ X = grads_stack for X using least squares
        result = torch.linalg.lstsq(A, grads_stack, driver='gels') # pylint:disable=not-callable
        X = result.solution  # Shape: (2, d)

        # Predict the next gradient
        next_T = n
        predicted_grad = X[0] * next_T + X[1]  # Shape: (d,)

        return predicted_grad