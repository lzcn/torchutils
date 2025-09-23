import unittest

import torch

from torchutils.ops import to as to_device
from tests.utils import get_available_device, requires_cuda


class TestToFunction(unittest.TestCase):
    def setUp(self):
        base = torch.randn(2, 3)
        self.data = {
            "tensor": base.clone(),
            "list": [base.clone(), "meta"],
            "tuple": (base.clone(), torch.tensor([1.0])),
            "string": "literal",
        }

    def test_to_cpu_preserves_structure(self):
        moved = to_device(self.data, device="cpu")

        self.assertEqual(moved["list"][1], "meta")
        self.assertIsInstance(moved["tuple"], tuple)
        for tensor in (moved["tensor"], moved["list"][0], moved["tuple"][0], moved["tuple"][1]):
            self.assertEqual(tensor.device.type, "cpu")

    def test_to_respects_available_device(self):
        device = get_available_device()
        moved = to_device(self.data, device)
        expected_type = device.type

        for tensor in (moved["tensor"], moved["list"][0], moved["tuple"][0], moved["tuple"][1]):
            self.assertEqual(tensor.device.type, expected_type)

    @requires_cuda
    def test_default_device_prefers_cuda(self):
        moved = to_device(self.data)

        for tensor in (moved["tensor"], moved["list"][0], moved["tuple"][0], moved["tuple"][1]):
            self.assertTrue(tensor.is_cuda)

    def test_to_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            to_device(42, "cpu")


if __name__ == "__main__":
    unittest.main()
