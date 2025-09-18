import functools
import os
from typing import Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable)


def get_rank() -> int:
    """Safely get the current global rank from environment variables.

    Returns:
        int: Global rank (defaults to 0 if not set).
    """
    for key in ("RANK", "LOCAL_RANK"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue
    return 0


def rank_zero_only(func: F) -> Optional[F]:
    """A decorator to ensure a function is only run on rank 0.

    Useful in distributed training for logging, saving models, etc.

    Args:
        func (Callable): The function to wrap.

    Returns:
        Callable: Wrapped function that only runs on rank 0.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            return func(*args, **kwargs)

    return wrapper  # type: ignore
