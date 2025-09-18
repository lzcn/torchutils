"""Logging utilities for distributed PyTorch training.

Provides rank-zero-only logging to prevent message duplication across GPU processes.
Works with standard Python logging, Hydra, and other configuration frameworks.

Example::

    from torchutils.logger import config, get_logger, register_formatter

    # Basic usage
    config(level="INFO", log_file="app.log")
    logger = get_logger(__name__)
    logger.info("Training started")

    # Custom formatter
    register_formatter("custom", {
        "format": "%(asctime)s: %(message)s",
        "datefmt": "%H:%M:%S"
    })
    config(formatter="custom")
"""

import logging

from torchutils.distributed import rank_zero_only


NAMED_FORMATTERS = {
    "default": {
        "format": "[%(levelname)s] - %(asctime)s - [%(name)s.%(funcName)s:%(lineno)d]: %(message)s",
        "datefmt": "%m-%d %H:%M:%S",
    },
    "simple": {
        "format": "[%(levelname)s] - %(asctime)s - [%(name)s]: %(message)s",
        "datefmt": "%m-%d %H:%M:%S",
    },
    "concise": {
        "format": "%(asctime)s: %(message)s",
        "datefmt": "%m-%d %H:%M:%S",
    },
}


def register_formatter(name, formatter):
    """Register a custom formatter.

    Args:
        name: Formatter name.
        formatter: Dict with 'format' and 'datefmt' keys.

    Example::

        register_formatter("my_format", {
            "format": "%(asctime)s: %(message)s",
            "datefmt": "%H:%M:%S"
        })
        config(formatter="my_format")
    """
    NAMED_FORMATTERS[name] = formatter


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

    # Ensure all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level_name in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level_name, rank_zero_only(getattr(logger, level_name)))

    return logger


def config(
    level="INFO",
    stream_level=None,
    file_level=None,
    log_file=None,
    file_mode="a",
    formatter="default",
    file_formatter=None,
    stream_formatter=None,
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
        formatter: Formatter name. One of: "default", "simple", "concise".
        file_formatter: File formatter. Uses formatter if None.
        stream_formatter: Console formatter. Uses formatter if None.

    Formatters:
        - "default": ``[LEVEL] - MM-DD HH:MM:SS - [name.function:line]: message``
        - "simple": ``[LEVEL] - MM-DD HH:MM:SS - [name]: message``
        - "concise": ``MM-DD HH:MM:SS: message``

    Example::

        config(level="INFO", log_file="app.log", formatter="simple")
        logger = get_logger(__name__)
    """
    from logging.config import dictConfig

    # Handle defaults
    file_level = file_level or level
    stream_level = stream_level or level
    file_formatter = file_formatter or formatter
    stream_formatter = stream_formatter or formatter

    # Validate formatter exists
    if formatter not in NAMED_FORMATTERS:
        raise ValueError(f"Unknown formatter: {formatter}. Available: {list(NAMED_FORMATTERS.keys())}")

    # configure stream handler
    stream_handler = {
        "class": "logging.StreamHandler",
        "formatter": stream_formatter,
        "level": stream_level,
    }
    # configure file handler
    file_handler = {
        "class": "logging.FileHandler",
        "formatter": file_formatter,
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
            "formatters": NAMED_FORMATTERS,
            "handlers": handlers,
            "root": {"level": "DEBUG", "handlers": handlers.keys()},
        }
    )
