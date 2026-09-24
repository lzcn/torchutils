import pytest
import torch
import torch.nn as nn

import torchutils as tu


def _model():
    return nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))


def test_feature_hook_captures_and_cleans():
    model = _model()
    with tu.FeatureHook(model, ["0", "1"]) as feats:
        model(torch.zeros(1, 2))
    assert feats["0"].shape == (1, 2)
    assert not model._forward_hooks


def test_failed_enter_does_not_leak_hooks():
    model = _model()
    with pytest.raises(AttributeError):
        with tu.FeatureHook(model, ["0", "nope"]):
            pass
    assert not model._forward_hooks


def test_grad_hook_captures():
    model = _model()
    with tu.GradHook(model, ["0"]) as grads:
        model(torch.zeros(1, 2)).sum().backward()
    assert grads["0"].shape == (1, 2)
