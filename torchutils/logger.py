"""Logging setup for PyTorch training, with rank-zero deduplication.

Example::

    import torchutils as tu

    tu.setup_logger(level="INFO", log_file="train.log")
    logging.getLogger(__name__).info("Emitted only on rank 0")
"""

import logging

from .distributed import _rank

__all__ = ["setup_logger"]

_DEFAULT_FORMAT = (
    "[%(levelname)s] - %(asctime)s - [%(name)s.%(funcName)s:%(lineno)d]: %(message)s"
)
_DEFAULT_DATEFMT = "%m-%d %H:%M:%S"


class _RankZeroFilter(logging.Filter):
    """Drop log records on non-zero ranks."""

    def filter(self, record) -> bool:
        return _rank() == 0


def setup_logger(
    level="INFO",
    stream_level=None,
    file_level=None,
    log_file=None,
    file_mode="a",
    format_string=None,
    date_format=None,
):
    """Configure the root logger with a console handler and optional file handler.

    Safe to call multiple times (handlers are reset on each call). Under
    distributed training only rank 0 emits records.

    Args:
        level: Default level for both handlers. Defaults to "INFO".
        stream_level: Console output level. Uses ``level`` if None.
        file_level: File output level. Uses ``level`` if None.
        log_file: Log file path. No file logging if None.
        file_mode: File mode ("a" or "w"). Defaults to "a".
        format_string: Custom format string. Uses default if None.
        date_format: Custom date format. Uses default if None.
    """
    formatter = logging.Formatter(
        format_string or _DEFAULT_FORMAT, datefmt=date_format or _DEFAULT_DATEFMT
    )

    stream_lvl = logging.getLevelName(stream_level or level)
    file_lvl = logging.getLevelName(file_level or level)

    root = logging.getLogger()
    root.setLevel(min(stream_lvl, file_lvl))
    root.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_lvl)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_RankZeroFilter())
    root.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(file_lvl)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_RankZeroFilter())
        root.addHandler(file_handler)
