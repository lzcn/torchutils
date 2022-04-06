import logging

import pytest

import torchutils

LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]

LOGGER = logging.getLogger("main")


def test_stream_logger(capsys):
    torchutils.logger.config()
    LOGGER.critical("")
    captured = capsys.readouterr()
    assert "main" in captured.err


def test_file_logger(tmp_path):
    d = tmp_path / "log"
    d.mkdir()
    p = d / "file.log"
    torchutils.logger.config(log_file=p)
    LOGGER.critical("")
    captured = p.read_text()
    assert "main" in captured


@pytest.mark.parametrize("level", LEVELS)
def test_logger_level(capsys, level):
    torchutils.logger.config(level=level)
    LOGGER.critical("")
    LOGGER.error("")
    LOGGER.warning("")
    LOGGER.info("")
    LOGGER.debug("")
    captured = capsys.readouterr()
    value = logging.getLevelName(level)
    for name in LEVELS:
        if logging.getLevelName(name) < value:
            assert name not in captured.err
        else:
            assert name in captured.err
