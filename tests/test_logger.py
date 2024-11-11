import logging
import unittest
import tempfile
import os
from unittest import mock

import torchutils

LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
LOGGER = logging.getLogger("main")


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
