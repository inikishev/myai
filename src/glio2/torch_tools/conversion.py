import typing as T
from collections import abc
import numpy as np
import torch

@T.overload
def maybe_ensure_detach_cpu(x: torch.Tensor) -> torch.Tensor: ...
@T.overload
def maybe_ensure_detach_cpu[X](x: X) -> X: ...
def maybe_ensure_detach_cpu[X](x: X) -> torch.Tensor | X:
    """If x is torch.tensor, ensures it is on CPU and detached. Otherwise returns x as is."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x

def ensure_numpy(x) -> np.ndarray:
    """Converts x to numpy array if it is not already one. If tensor, ensures it is detached and on CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x, copy=False)

@T.overload
def ensure_numpy_or_none(x: torch.Tensor) -> np.ndarray: ...
@T.overload
def ensure_numpy_or_none(x: None) -> None: ...
@T.overload
def ensure_numpy_or_none(x: np.ndarray) -> np.ndarray: ...
def ensure_numpy_or_none(x: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    """Converts x to numpy array if it is not already one. If tensor, ensures it is detached and on CPU. If None, returns None."""
    if x is None: return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x, copy=False)


@T.overload
def maybe_ensure_cpu_number(x: torch.Tensor | np.ndarray) -> np.ndarray | float | int: ...
@T.overload
def maybe_ensure_cpu_number(x: int) -> int: ...
@T.overload
def maybe_ensure_cpu_number(x: float) -> float: ...
@T.overload
def maybe_ensure_cpu_number[X](x: X) -> X: ...
def maybe_ensure_cpu_number[X](x: int | float | torch.Tensor | np.ndarray | X) -> np.ndarray | int | float | X:
    """Size one arrays and tensors are returned as python scalar types. Tensors are converted to numpy arrays. Anything else returned as is."""
    if isinstance(x, np.ndarray):
        if x.size == 1: return x.item()
        return x
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1: return x.item()
        return x.numpy()
    return x


@T.overload
def maybe_ensure_detach_cpu_recursive(x: torch.Tensor) -> torch.Tensor: ...
@T.overload
def maybe_ensure_detach_cpu_recursive[K](x: abc.Mapping[K, torch.Tensor | T.Any]) -> dict[K, torch.Tensor | T.Any]: ...
@T.overload
def maybe_ensure_detach_cpu_recursive(x: abc.MutableSequence[torch.Tensor | T.Any]) -> list[torch.Tensor | T.Any]: ...
@T.overload
def maybe_ensure_detach_cpu_recursive(x: tuple[torch.Tensor | T.Any]) -> tuple[torch.Tensor | T.Any]: ...
@T.overload
def maybe_ensure_detach_cpu_recursive[V](x: V) -> V: ...
def maybe_ensure_detach_cpu_recursive(x):
    """Recursively detaches and moves x or all elements in x to CPU. Can be slow!"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, abc.Mapping):
        return {k: maybe_ensure_detach_cpu_recursive(v) for k, v in x.items()}
    elif isinstance(x, abc.MutableSequence):
        return [maybe_ensure_detach_cpu_recursive(v) for v in x]
    elif isinstance(x, abc.Sequence):
        return tuple(maybe_ensure_detach_cpu_recursive(v) for v in x)
    return x

def ensure_numpy_recursive(x) -> np.ndarray:
    """Converts x to numpy array if it is not already one. If tensor, ensures it is detached and on CPU, RECURSIVELY. Can be slow!"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, abc.Sequence):
        return np.asarray([ensure_numpy_recursive(i) for i in x])
    else:
        return x

def ensure_numpy_or_none_recursive(x) -> np.ndarray | None:
    """Converts x to numpy array if it is not already one. If tensor, ensures it is detached and on CPU, RECURSIVELY. Can be slow! If None, returns None."""
    if x is None: return None
    return ensure_numpy_or_none_recursive(x)