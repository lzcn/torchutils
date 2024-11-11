import unittest
import torch
import torchutils


class TestBackboneLoading(unittest.TestCase):

    def test_backbone_loading(self):
        backbones = torchutils.backbones._BACKBONES.keys()
        for name in backbones:
            torchutils.backbone(name, weights=None)


if __name__ == "__main__":
    unittest.main()
