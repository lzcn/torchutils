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
    NAMED_FORMATTERS[name] = formatter


def get_value(x, default):
    return default if x is None else x


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
    """Logger configuration with handlers.

    All loggers pass their message to the root logger. Thus, this method configures all
    loggers by attaching handlers to the root logger.

    Predifned formatters:

        - ``"default"``: ``[LEVEL] - %m-%d %H:%M:%S - [name.function.line]: message``
        - ``"simple"``： ``[LEVEL] - %m-%d %H:%M:%S - [name]: message``
        - ``"concise"``： ``%m-%d %H:%M:%S: message``

    Args:
        level (str, optional):  default logging level for all handlers. Defaults to ``"INFO"``
        stream_level (str, optional): logging level for STDOUT. Defaults to ``level``
        file_level (str, optional): logging level for log file Defaults to ``level``.
        log_file (str, optional): log file. If given, file logger will be enabled. Defaults to ``None``.
        file_mode (str, optional): log file mode. Defaults to ``"a"``.
        formatter (str, optional): message format. Defaults to ``"default"``.
        file_formatter (str, optional): message format for stream. Defaults to ``formatter``.
        stream_formatter (str, optional): message format for file. Defaults to ``formatter``.

    """
    from logging.config import dictConfig

    file_level = get_value(file_level, level)
    file_formatter = get_value(file_formatter, formatter)
    stream_level = get_value(stream_level, level)
    stream_formatter = get_value(stream_formatter, formatter)

    # configure stream logger
    stream_hanlder = {
        "class": "logging.StreamHandler",
        "formatter": stream_formatter,
        "level": stream_level,
    }
    # configure file logger
    file_handler = {
        "class": "logging.FileHandler",
        "formatter": file_formatter,
        "level": file_level,
        "filename": log_file,
        "mode": file_mode,
    }
    if log_file is None:
        handlers = {"stream": stream_hanlder}
    else:
        handlers = {"stream": stream_hanlder, "file": file_handler}

    dictConfig(
        {
            "version": 1.0,
            "disable_existing_loggers": False,
            "formatters": NAMED_FORMATTERS,
            "handlers": handlers,
            "root": {"level": "DEBUG", "handlers": handlers.keys()},
        }
    )
