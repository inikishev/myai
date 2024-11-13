import typing as T

def func2func[**P, R](wrapper: T.Callable[P, R]):
    """Copies the signature from one function to another. Works with VSCode autocomplete."""

    def decorator(func: T.Callable) -> T.Callable[P, R]:
        func.__doc__ = wrapper.__doc__
        return func

    return decorator

def func2method[**P, R](wrapper: T.Callable[P, R]):
    """Copies the signature a function to a method. Works with VSCode autocomplete."""

    def decorator(func: T.Callable) -> T.Callable[T.Concatenate[T.Any, P], R]:
        func.__doc__ = wrapper.__doc__
        return func

    return decorator

def method2method[**P, R](wrapper: T.Callable[T.Concatenate[T.Any, P], R]):
    """Copies the signature from a method to a method. Works with VSCode autocomplete."""

    # the T.Any here is the self argument.
    def decorator(func: T.Callable[T.Concatenate[T.Any, T.Any, P], R]) -> T.Callable[T.Concatenate[T.Any, P], R]:
        func.__doc__ = wrapper.__doc__
        return func # type:ignore

    return decorator

def method2method_return_override[**P, R, RNew](wrapper: T.Callable[T.Concatenate[T.Any, P], R], ret: T.Type[RNew]):
    """Copies the signature from a method to a method, overrides return with the type specified in `ret`. Works with VSCode autocomplete."""

    # the T.Any here is the self argument.
    def decorator(func: T.Callable[T.Concatenate[T.Any, T.Any, P], R]) -> T.Callable[T.Concatenate[T.Any, P], RNew]:
        func.__doc__ = wrapper.__doc__
        return func # type:ignore

    return decorator

def method2func[**P, R](wrapper: T.Callable[T.Concatenate[T.Any, P], R]):
    """Copies the signature from a method to a function. Works with VSCode autocomplete."""

    def decorator(func: T.Callable[T.Concatenate[T.Any, P], R]) -> T.Callable[P, R]:
        func.__doc__ = wrapper.__doc__
        return func # type:ignore

    return decorator
