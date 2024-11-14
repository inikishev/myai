
import inspect
import typing as T
from collections.abc import Callable, Iterable

from .iterables import flatten
from .identity import identity

type Composable[**P, R] = Callable[P, R] | Iterable[Callable[P, R]]

class Compose:
    """Compose multiple functions into a single function. Note that functions will be flattened."""
    def __init__(self, *functions: Composable):
        self.functions = flatten(functions)

    def __call__(self, x):
        for t in self.functions:
            x = t(x)
        return x

    def __add__(self, other: Composable):
        return Compose(*self.functions, other)

    def __radd__(self, other: Composable):
        return Compose(other, *self.functions)

    def __str__(self):
        return f"Compose({', '.join(str(t) for t in self.functions)})"

    def __iter__(self):
        return iter(self.functions)

    def __getitem__(self, i): return self.functions[i]
    def __setitem__(self, i, v): self.functions[i] = v
    def __delitem__(self, i): del self.functions[i]

def compose(*functions: Composable) -> Callable:
    flattened = flatten(functions)
    if len(flattened) == 1: return flattened[0]
    return Compose(*flattened)

def maybe_compose(*functions: Callable | None | Iterable[Callable | None]) -> Callable:
    """Compose some functions while ignoring None, if got only None, returns identity."""
    flattened = [i for i in flatten(functions) if i is not None]
    if len(flattened) == 1: return flattened[0]
    if len(flattened) == 0: return identity
    return Compose(*flattened)

def get_full_kwargs[**P2](fn: Callable[P2, T.Any], *args: P2.args, **kwargs: P2.kwargs) -> dict[str, T.Any]:
    """Returns a dictionary of all keyword arguments of a function called with given args and kwargs."""
    sig = inspect.signature(fn).bind(*args, **kwargs)
    sig.apply_defaults()
    return sig.arguments

class SaveSignature:
    def __init__[**K](self, obj: Callable[K, T.Any], *args: K.args, **kwargs: K.kwargs):
        self.obj: Callable = obj
        self.signature = get_full_kwargs(obj, *args, **kwargs)