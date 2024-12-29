import logging
import os
import tempfile
import unittest
from unittest import mock

import torchutils
from torchutils.logger import get_logger

LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
LOGGER = logging.getLogger("main")


class TestGetLogger(unittest.TestCase):

    def setUp(self):
        self.logger_name = "test_logger"
        self.logger = get_logger(self.logger_name, level=logging.INFO)

    def test_logger_initialization(self):
        self.assertEqual(self.logger.name, self.logger_name)
        self.assertEqual(self.logger.level, logging.INFO)

    def test_rank_zero_only_decorator(self):
        # Test when LOCAL_RANK is 0
        os.environ["LOCAL_RANK"] = "0"
        with self.assertLogs(self.logger, level="INFO") as log:
            self.logger.info("This should be logged")
        self.assertIn("INFO:test_logger:This should be logged", log.output)

        # Test when LOCAL_RANK is not 0, AssertionError should be raised
        os.environ["LOCAL_RANK"] = "1"
        with self.assertRaises(AssertionError):
            with self.assertLogs(self.logger, level="INFO") as log:
                self.logger.info("This should not be logged")


class TestLogger(unittest.TestCase):
    def setUp(self):
        torchutils.logger.config()

    @mock.patch("sys.stderr")
    def test_stream_logger(self, mock_stderr):
        LOGGER.critical("")
        mock_stderr.getvalue = lambda: "main"
        self.assertIn("main", mock_stderr.getvalue())

    def test_file_logger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "file.log")
            torchutils.logger.config(log_file=log_path)
            LOGGER.critical("")
            with open(log_path, "r") as f:
                content = f.read()
            self.assertIn("main", content)

    def test_logger_level(self):
        for level in LEVELS:
            with self.subTest(level=level):
                torchutils.logger.config(level=level)
                with self.assertLogs("main", level=level) as cm:
                    LOGGER.critical("")
                    LOGGER.error("")
                    LOGGER.warning("")
                    LOGGER.info("")
                    LOGGER.debug("")
                captured = cm.output
                value = getattr(logging, level)
                for name in LEVELS:
                    if getattr(logging, name) < value:
                        self.assertNotIn(name, "".join(captured))
                    else:
                        self.assertIn(name, "".join(captured))


if __name__ == "__main__":
    unittest.main()
