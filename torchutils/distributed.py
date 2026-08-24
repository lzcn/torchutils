from collections.abc import Callable
import functools
import os
from typing import TypeVar

import torch.distributed as dist

F = TypeVar("F", bound=Callable)

__all__ = ["rank_zero_only"]


def _rank() -> int:
    """Current global rank: torch.distributed if initialized, else the RANK env var."""
    if dist.is_initialized():
        return dist.get_rank()
    try:
        return int(os.environ.get("RANK", 0))
    except ValueError:
        return 0


def rank_zero_only(func: F) -> F:
    """Decorator: run the function only on global rank 0 (no-op elsewhere)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _rank() == 0:
            return func(*args, **kwargs)

    return wrapper
