import os
import unittest

from torchutils.distributed import rank_zero_only


class TestRankZeroOnly(unittest.TestCase):

    def test_rank_zero(self):
        @rank_zero_only
        def test_func():
            return "Executed"

        os.environ["LOCAL_RANK"] = "0"
        self.assertEqual(test_func(), "Executed")

    def test_non_rank_zero(self):
        @rank_zero_only
        def test_func():
            return "Executed"

        os.environ["LOCAL_RANK"] = "1"
        self.assertIsNone(test_func())


if __name__ == "__main__":
    unittest.main()

