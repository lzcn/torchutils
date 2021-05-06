NAMED_FORMATTERS = {
    "default": {
        "format": "[%(levelname)s] - %(asctime)s - [%(name)s.%(funcName)s:%(lineno)d]: %(message)s",
        "datefmt": "%m-%d %H:%M:%S",
    },
    "simple": {
        "format": "[%(levelname)s] - %(asctime)s - [%(name)s]: %(message)s",
        "datefmt": "%m-%d %H:%M:%S",
    },
}


def reigister_formatter(name, formatter):
    NAMED_FORMATTERS[name] = formatter


def config(
    name="main",
    stream_level="INFO",
    file_level="INFO",
    log_file=None,
    file_mode="a",
    formatter="default",
    file_formatter=None,
    stream_formatter=None,
):
    """Config the loggers.


    There are two formats for messages:

    - "default": [level] - time - [name.function.line]: message
    - "simple"： [level] - time - [name]: message

    Args:
        stream_level (str, optional): logging level for STDOUT. Defaults to ``"INFO"``
        file_level (str, optional): logging level for log file Defaults to ``"INFO"``.
        log_file (str, optional): log file. If it is not given, then disable logging to
            file. Defauts to ``None``.
        file_mode (str, optional): log file mode. Defaults to "a".
        formatter (str, optional): message format. Defaults to ``"default"``.
        file_formatter (str, optional): message format for stream. Defaults to ``formatter``.
        stream_formatter (str, optional): message format for file. Defaults to ``formatter``.

    """
    import logging
    from logging.config import dictConfig

    file_formatter = formatter if file_formatter is None else file_formatter
    stream_formatter = formatter if stream_formatter is None else stream_formatter
    # get stream logger
    stream_hanlder = {
        "class": "logging.StreamHandler",
        "formatter": stream_formatter,
        "level": stream_level,
    }
    # get file logger
    if log_file is None:
        file_handler = {"class": "logging.NullHandler"}
    else:
        file_handler = {
            "class": "logging.FileHandler",
            "formatter": file_formatter,
            "level": file_level,
            "filename": log_file,
            "mode": file_mode,
        }

    dictConfig(
        {
            "version": 1.0,
            "disable_existing_loggers": False,
            "formatters": NAMED_FORMATTERS,
            "handlers": {"stream": stream_hanlder, "file": file_handler},
            "root": {"level": "DEBUG", "handlers": ["stream", "file"]},
        }
    )
    logger = logging.getLogger(name)
    return logger
