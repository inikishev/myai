import numpy as np
import torch

def tonumpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x, copy=False)

def totensor(x, dtype = None, device = None):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype, copy=False)
    elif isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, copy=False)
    return torch.tensor(x, device=device, dtype=dtype,)