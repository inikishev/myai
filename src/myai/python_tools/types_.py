from collections import abc
import typing as T

_T_co = T.TypeVar("_T_co", covariant=True)

class SupportsNext(T.Protocol[_T_co]):
    def __next__(self) -> _T_co: ...

class HasIterDunder(T.Protocol[_T_co]):
    def __iter__(self) -> _T_co: ...

class SupportsLenAndGetitem(T.Protocol[_T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, __k: int, /) -> _T_co: ...

type SupportsIter[_T_co] = HasIterDunder[_T_co] | SupportsLenAndGetitem[_T_co]
