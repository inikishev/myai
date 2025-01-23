#type:ignore
import torch
from torch.optim import Optimizer

class PertubationConsensus(Optimizer):
    def __init__(self, params, lr=1e-3, sigma = 1e-1, num_grads=3, discount=0.1):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 1 <= num_grads:
            raise ValueError(f"Invalid num_grads: {num_grads}")
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f"Invalid discount factor: {discount}")

        defaults = dict(lr=lr, num_grads=num_grads, discount=discount)
        self.sigma = sigma
        super().__init__(params, defaults)

        self.state['step'] = 0

    @torch.no_grad
    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required for SetTheoryOptimizer")

        # Initialize storage for gradients across multiple passes
        grad_storage = []
        param_groups = self.param_groups
        params = []
        for group in param_groups:
            for p in group['params']:
                grad_storage.append([])
                params.append(p)

        # Perform K forward and backward passes to collect gradients
        num_grads = param_groups[0]['num_grads']
        for k in range(num_grads):
            old_params = [p.clone() for p in params]
            for p in params:
                mean = p.abs().mean()
                if mean > 0:
                    p.add_(torch.randn_like(p), alpha = self.sigma*mean)

            self.zero_grad()
            with torch.enable_grad(): loss = closure()
            # loss.backward(retain_graph=(k < num_grads - 1))

            # Capture current gradients
            idx = 0
            for group in param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        grad_storage[idx].append(p.grad.detach().clone())
                    else:
                        grad_storage[idx].append(torch.zeros_like(p.data))
                    idx += 1

            for o, p in zip(old_params, params):
                p.set_(o)

        # Process gradients and update parameters
        idx = 0
        for group in param_groups:
            lr = group['lr']
            discount = group['discount']
            for p in group['params']:
                if p.grad is None:
                    idx += 1
                    continue

                grads = grad_storage[idx]
                stacked_grads = torch.stack(grads)  # [num_grads, *param_shape]

                # Determine consensus mask (all grads same sign)
                signs = torch.sign(stacked_grads)
                all_positive = (signs == 1).all(dim=0)
                all_negative = (signs == -1).all(dim=0)
                consensus_mask = all_positive | all_negative

                # Compute average gradient
                avg_grad = stacked_grads.mean(dim=0)

                # Apply discount where no consensus
                final_grad = torch.where(consensus_mask, avg_grad, avg_grad * discount)

                # Update parameters
                p.data.add_(final_grad, alpha=-lr)

                idx += 1

        self.state['step'] += 1
        return loss