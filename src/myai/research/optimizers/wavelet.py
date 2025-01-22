import torch
from torch.optim import Optimizer

def haar_transform(x):
    x = x.view(-1)
    n = x.size(0)
    if n % 2 != 0:
        x = torch.cat([x, torch.zeros(1, dtype=x.dtype, device=x.device)])
    even = x[::2]
    odd = x[1::2]
    avg = (even + odd) / 2
    diff = (even - odd) / 2
    return torch.cat([avg, diff])

def inverse_haar_transform(coeffs):
    n = coeffs.size(0)
    if n % 2 != 0:
        raise ValueError("Coefficients must be even length")
    mid = n // 2
    avg = coeffs[:mid]
    diff = coeffs[mid:]
    even = avg + diff
    odd = avg - diff
    reconstructed = torch.zeros(n, dtype=coeffs.dtype, device=coeffs.device)
    reconstructed[::2] = even
    reconstructed[1::2] = odd
    return reconstructed

class WaveletOptimizer(Optimizer):
    """this flattens the gradients and applies wavelet transform on them. And the todo is to not flatten them. Can't be bothered."""
    def __init__(self, params, lr=1e-3, low_ratio=0.5, high_ratio=1.5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if low_ratio < 0 or high_ratio < 0:
            raise ValueError("Scaling ratios must be non-negative")
        defaults = dict(lr=lr, low_ratio=low_ratio, high_ratio=high_ratio)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            low_ratio = group['low_ratio']
            high_ratio = group['high_ratio']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("WaveletOptimizer does not support sparse gradients")

                original_shape = grad.shape
                flattened_grad = grad.flatten().clone()  # Clone to avoid modifying original grad
                n_original = flattened_grad.size(0)

                # Pad to even length if necessary
                padded = False
                if n_original % 2 != 0:
                    flattened_grad = torch.cat([flattened_grad, torch.zeros(1, device=flattened_grad.device)])
                    padded = True

                # Apply Haar transform
                coeffs = haar_transform(flattened_grad)
                mid = coeffs.size(0) // 2
                approx = coeffs[:mid]
                details = coeffs[mid:]

                # Scale coefficients
                scaled_approx = approx * low_ratio
                scaled_details = details * high_ratio
                modified_coeffs = torch.cat([scaled_approx, scaled_details])

                # Inverse transform
                modified_grad_flat = inverse_haar_transform(modified_coeffs)

                # Truncate if padded
                if padded:
                    modified_grad_flat = modified_grad_flat[:n_original]

                # Reshape and apply learning rate
                modified_grad = modified_grad_flat.reshape(original_shape) * lr

                # Update parameter
                p.data.sub_(modified_grad)

        return loss