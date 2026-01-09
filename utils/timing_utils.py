import functools
from typing import Callable
import time
from config import DEBUG_TIMING

def timeit(name: str | None = None):
    """
    Decorator to measure execution time of a function.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not DEBUG_TIMING:
                return func(*args, **kwargs)
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()

            fname = name if name else func.__qualname__
            print(f"⏱️ {fname} took {(end - start)*1000:.2f} ms")

            return result
        return wrapper
    return decorator