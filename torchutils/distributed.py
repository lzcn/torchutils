import functools
import os
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def get_rank() -> int:
    """Get the current global rank from environment variables.

    Returns:
        Global rank (defaults to 0 if not in distributed mode).
    """
    for key in ("RANK", "LOCAL_RANK"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue
    return 0


def rank_zero_only(func: F) -> F:
    """Decorator to ensure a function only runs on rank 0.

    Useful in distributed training for logging, saving models, etc.

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function that only runs on rank 0.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            return func(*args, **kwargs)

    return wrapper
