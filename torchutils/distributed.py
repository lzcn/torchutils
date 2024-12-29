import os
import functools


def rank_zero_only(func):
    """
    Decorator that ensures the wrapped function is only called on rank 0.

    Args:
        func (Callable): The function to be wrapped.

    Returns:
        Callable: The wrapped function.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            return func(*args, **kwargs)

    return wrapper
