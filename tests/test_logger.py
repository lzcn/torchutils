import logging

import pytest

import torchutils as tu


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        tu.setup_logger(level="FLOOD")


def test_file_handler_writes(tmp_path):
    tu.setup_logger(level="INFO", log_file=tmp_path / "train.log", file_mode="w")
    logging.getLogger("torchutils.test").info("hello")
    assert "hello" in (tmp_path / "train.log").read_text()
