"""Logging utilities for distributed PyTorch training.

Provides rank-zero-only logging to prevent message duplication across GPU processes.

Example::

    import torchutils as tu

    tu.setup_logger(level="INFO", log_file="app.log")
    logger = tu.get_logger(__name__)
    logger.info("Training started")
"""

import logging

from .distributed import rank_zero_only

# Single default formatter
_DEFAULT_FORMATTER = {
    "format": "[%(levelname)s] - %(asctime)s - [%(name)s.%(funcName)s:%(lineno)d]: %(message)s",
    "datefmt": "%m-%d %H:%M:%S",
}


def get_logger(name=__name__, level=None):
    """Get a rank-zero-only logger for distributed training.

    Returns a logger that only outputs messages on rank 0, preventing
    log duplication in multi-GPU training. Does not configure handlers
    or levels - use config() or external frameworks for that.

    Args:
        name: Logger name. Defaults to caller's module name.
        level: Logger level. If provided, sets the logger level.

    Returns:
        Rank-zero-only logger instance.

    Example::

        logger = get_logger(__name__)
        logger.info("Only rank 0 prints this")
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    # Apply rank_zero_only to all logging methods
    for level_name in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level_name, rank_zero_only(getattr(logger, level_name)))

    return logger


def config(
    level="INFO",
    stream_level=None,
    file_level=None,
    log_file=None,
    file_mode="a",
    format_string=None,
    date_format=None,
):
    """Configure root logger with handlers and formatters.

    Sets up console and optional file logging. Call once at startup,
    then use get_logger() to obtain logger instances.

    Args:
        level: Default level for all handlers. Defaults to "INFO".
        stream_level: Console output level. Uses level if None.
        file_level: File output level. Uses level if None.
        log_file: Log file path. No file logging if None.
        file_mode: File mode ("a" or "w"). Defaults to "a".
        format_string: Custom format string. Uses default if None.
        date_format: Custom date format. Uses default if None.

    Example::

        config(level="INFO", log_file="app.log")
        logger = get_logger(__name__)
    """
    from logging.config import dictConfig

    # Handle defaults
    file_level = file_level or level
    stream_level = stream_level or level

    # Build formatter
    formatter_config = {
        "format": format_string or _DEFAULT_FORMATTER["format"],
        "datefmt": date_format or _DEFAULT_FORMATTER["datefmt"],
    }

    # Configure handlers
    stream_handler = {
        "class": "logging.StreamHandler",
        "formatter": "default",
        "level": stream_level,
    }

    file_handler = {
        "class": "logging.FileHandler",
        "formatter": "default",
        "level": file_level,
        "filename": log_file,
        "mode": file_mode,
    }

    if log_file is None:
        handlers = {"stream": stream_handler}
    else:
        handlers = {"stream": stream_handler, "file": file_handler}

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"default": formatter_config},
            "handlers": handlers,
            "root": {"level": "DEBUG", "handlers": handlers.keys()},
        }
    )
