import logging
import re

from colorama import Back, Fore, Style, init

# Initialize Colorama
init(autoreset=True)


TAGS_TO_COLORS = {
    "{red}": Fore.RED,
    "{green}": Fore.GREEN,
    "{yellow}": Fore.YELLOW,
    "{blue}": Fore.BLUE,
    "{magenta}": Fore.MAGENTA,
    "{cyan}": Fore.CYAN,
    "{white}": Fore.WHITE,
    "{black}": Fore.BLACK,
    "{bg_red}": Back.RED,
    "{bg_green}": Back.GREEN,
    "{bg_yellow}": Back.YELLOW,
    "{bg_blue}": Back.BLUE,
    "{bg_magenta}": Back.MAGENTA,
    "{bg_cyan}": Back.CYAN,
    "{bg_white}": Back.WHITE,
    "{bg_black}": Back.BLACK,
    "{bright}": Style.BRIGHT,
    "{dim}": Style.DIM,
    "{normal}": Style.NORMAL,
    "{reset}": Style.RESET_ALL,
}


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


# Custom formatter for console that uses Colorama for colors
class ColoramaFormatter(logging.Formatter):

    def format(self, record):
        message = super().format(record)
        # Replace custom tags with Colorama colors
        for tag, color in TAGS_TO_COLORS.items():
            message = message.replace(tag, color)
        return message


# Custom formatter for file that removes color tags
class ColoramaFileFormatte(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        # Remove custom tags
        message = re.sub(r"\{\/?[\w]+\}", "", message)
        return message


def stream_formatter_factory(formatter_name="default"):
    formatter_config = NAMED_FORMATTERS.get(formatter_name, NAMED_FORMATTERS["default"])
    # Check if the formatter_name exists in NAMED_FORMATTERS, else use "default"
    # Instantiate ColoramaFormatter with the specified or default format and datefmt
    return ColoramaFormatter(fmt=formatter_config["format"], datefmt=formatter_config["datefmt"])


def file_formatter_factory(formatter_name="default"):
    # Check if the formatter_name exists in NAMED_FORMATTERS, else use "default"
    formatter_config = NAMED_FORMATTERS.get(formatter_name, NAMED_FORMATTERS["default"])
    # Instantiate FileFormatter with the specified or default format and datefmt
    return ColoramaFileFormatte(fmt=formatter_config["format"], datefmt=formatter_config["datefmt"])


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
    color_tags=False,
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
        color_tas (bool, optional): whether allow color tags in log messages. Defaults to ``False``.

    """
    from logging.config import dictConfig

    file_level = get_value(file_level, level)
    file_formatter = get_value(file_formatter, formatter)
    stream_level = get_value(stream_level, level)
    stream_formatter = get_value(stream_formatter, formatter)

    if color_tags:
        # add colored formatter
        formatters = {
            "stream": {"()": stream_formatter_factory, "formatter_name": stream_formatter},
            "file": {"()": file_formatter_factory, "formatter_name": file_formatter},
        }
    else:
        formatters = NAMED_FORMATTERS

    # configure stream logger
    stream_hanlder = {
        "class": "logging.StreamHandler",
        "formatter": stream_formatter if not color_tags else "stream",
        "level": stream_level,
    }
    # configure file logger
    file_handler = {
        "class": "logging.FileHandler",
        "formatter": file_formatter if not color_tags else "file",
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
            "formatters": formatters,
            "handlers": handlers,
            "root": {"level": "DEBUG", "handlers": handlers.keys()},
        }
    )
