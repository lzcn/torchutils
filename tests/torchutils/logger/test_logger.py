import pytest
import torchutils
import logging

LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


def test_logger_name(capsys):
    name = "testing"
    logger = torchutils.logger.config(name=name)
    logger.critical("")
    logger.error("")
    logger.warning("")
    logger.info("")
    logger.debug("")
    captured = capsys.readouterr()
    assert name in captured.err


@pytest.mark.parametrize("stream_level", LEVELS)
def test_logger_level(capsys, stream_level):
    logger = torchutils.logger.config(stream_level=stream_level)
    logger.critical("")
    logger.error("")
    logger.warning("")
    logger.info("")
    logger.debug("")
    captured = capsys.readouterr()
    level = logging.getLevelName(stream_level)
    for name in LEVELS:
        if logging.getLevelName(name) < level:
            assert name not in captured.err
        else:
            assert name in captured.err
