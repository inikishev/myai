import joblib
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from ..data import DS

MEAN = 33.3643
STD = 78.5836

normalize = v2.Normalize([MEAN], [STD])

def _download_and_create_mnist(root):
    train = MNIST(root, train=True, download = True)
    test = MNIST(root, train=False, download = True)

    loader = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32), normalize])

    ds = DS()
    ds.add_samples_(train, loader = loader)
    ds.add_samples_(test, loader = loader)

    return ds

def get_MNIST(path = r"D:\datasets\MNIST.joblib") -> DS[tuple[torch.Tensor, int]]:
    """Returns entire MNIST classification DS with 70,000 samples, z-normalized.
    Each sample is a tuple with (1x28x28) tensor and a class index."""
    return joblib.load(path)