import contextlib
import typing as T
import time

@contextlib.contextmanager
def perf_counter_context(name: T.Optional[str | T.Any] = None, ndigits: T.Optional[int] = None):
    time_start = time.perf_counter()
    yield
    time_took = time.perf_counter() - time_start
    if name is None: name = "Context"
    if ndigits is not None: time_took = round(time_took, ndigits)
    print(f"{name} took {time_took} perf_counter seconds")

@contextlib.contextmanager
def time_context(name: T.Optional[str | T.Any] = None, ndigits: T.Optional[int] = None):
    time_start = time.time()
    yield
    time_took = time.time() - time_start
    if name is None: name = "Context"
    if ndigits is not None: time_took = round(time_took, ndigits)
    print(f"{name} took {time_took} perf_counter seconds")


class PerfCounter:
    def __init__(self):
        self.times= []

    def step(self):
        self.times.append(time.perf_counter())