from collections import abc
import typing as T

_T_co = T.TypeVar("_T_co", covariant=True)

class _HasNextDunder(T.Protocol[_T_co]):
    def __next__(self) -> _T_co: ...

class _HasIterDunder(T.Protocol[_T_co]):
    def __iter__(self) -> _T_co: ...

class _HasLenAndGetitemDunder(T.Protocol[_T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, k: int, /) -> _T_co: ...

type SupportsIter[_T_co] = _HasIterDunder[_T_co] | _HasIterDunder[_T_co] | _HasLenAndGetitemDunder[_T_co]