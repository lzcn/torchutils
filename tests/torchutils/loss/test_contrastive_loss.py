import pytest
import torch

import torchutils


def test_random():
    x = torch.randn(100, 10)
    loss = torchutils.loss.contrastive_loss(x, x, margin=0.0, reduction="mean")
    assert loss == pytest.approx(0.0)


def test_mask():
    x = torch.randn(100, 10)
    y = torch.randn(100, 10)
    mask = torch.randn(100) > 0.0
    while mask.sum() < 2:
        mask = torch.randn(100) > 0.0
    loss = torchutils.loss.contrastive_loss(x, y, margin=0.0, mask=mask, reduction="none")
    for flag, value in zip(mask, loss):
        if not flag:
            assert value == pytest.approx(0.0)


def test_reduction():
    x = torch.randn(100, 10)
    y = torch.randn(100, 10)
    loss_mean = torchutils.loss.contrastive_loss(x, y, reduction="mean")
    loss_none = torchutils.loss.contrastive_loss(x, y, reduction="none")
    loss_sum = torchutils.loss.contrastive_loss(x, y, reduction="sum")
    assert loss_none.sum() == pytest.approx(loss_sum.item())
    assert loss_none.mean() == pytest.approx(loss_mean.item())


def test_reduction_with_mask():
    x = torch.randn(100, 10)
    y = torch.randn(100, 10)
    mask = torch.randn(100) > 0.0
    while mask.sum() < 2:
        mask = torch.randn(100) > 0.0
    loss_mean = torchutils.loss.contrastive_loss(x, y, mask=mask, reduction="mean")
    loss_none = torchutils.loss.contrastive_loss(x, y, mask=mask, reduction="none")
    loss_sum = torchutils.loss.contrastive_loss(x, y, mask=mask, reduction="sum")
    assert (loss_none * mask).sum() == pytest.approx(loss_sum.item())
    assert (loss_none.sum() / mask.sum()) == pytest.approx(loss_mean.item())
