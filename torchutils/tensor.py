"""Tensor helpers and legacy wrappers."""

import torch

from ._internal import set_module


@set_module("torchutils")
def one_hot(index, num):
    """Convert the index tensor to one-hot encoding (deprecated)."""

    index = index.view(-1, 1)
    one_hot = torch.zeros(index.numel(), num).to(index.device)
    return one_hot.scatter_(1, index, 1.0)
