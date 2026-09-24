from collections import namedtuple

import torch

import torchutils as tu


def test_nested_structure_moved():
    batch = {"x": torch.zeros(1), "y": [torch.ones(1), (torch.full((1,), 2.0),)], "z": "meta"}
    out = tu.to(batch, "cpu", non_blocking=False)
    assert out["z"] == "meta"
    assert out["x"].device.type == "cpu"
    assert out["y"][1][0].item() == 2.0


def test_namedtuple_preserved():
    Point = namedtuple("Point", ["x", "y"])
    p = tu.to(Point(torch.zeros(1), torch.ones(1)), "cpu")
    assert isinstance(p, Point)
    assert p.x.item() == 0.0 and p.y.item() == 1.0


def test_device_object_accepted():
    t = tu.to(torch.zeros(1), torch.device("cpu"))
    assert t.device.type == "cpu"
