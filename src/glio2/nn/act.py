# pylint:disable=not-callable
import torch

class I1e(torch.nn.Module):
    """This worked really well in a single image autoencoding so I need to test this. It's also definitely quite slow."""
    def forward(self, x: torch.Tensor): return torch.special.i1e(x)