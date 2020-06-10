def config(stream_level="INFO", file_level="INFO", log_file=None, formatter="default"):
    """Config the loggers.


    There are two formats for messages:

    - "default": [level] - time - [name.function.line]: message
    - "simple"： [level] - time - [name]: message

    Args:
        stream_level (str, optional): logging level for STDOUT. Defaults to ``"INFO"``
        file_level (str, optional): logging level for log file Defaults to ``"INFO"``.
        log_file (str, optional): log file. If it is not given, then disable logging to
            file. Defauts to ``None``.
        formatter (str, optional): message format. Defaults to ``"default"``.

    """
    from logging.config import dictConfig

    stream_hanlder = {
        "class": "logging.StreamHandler",
        "formatter": formatter,
        "level": stream_level
    }
    if log_file is None:
        file_handler = {"class": "logging.NullHandler"}
    else:
        file_handler = {
            "class": "logging.FileHandler",
            "formatter": formatter,
            "level": file_level,
            "filename": log_file,
        }

    dictConfig(
        {
            "version": 1.0,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "[%(levelname)s] - %(asctime)s - [%(name)s.%(funcName)s:%(lineno)d]: %(message)s",
                    "datefmt": "%m-%d %H:%M:%S",
                },
                "simple": {
                    "format": "[%(levelname)s] - %(asctime)s - [%(name)s]: %(message)s",
                    "datefmt": "%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "stream": stream_hanlder,
                "file": file_handler,
            },
            "root": {
                "level": "DEBUG",
                "handlers": ["stream", "file"],
            },
        }
    )
