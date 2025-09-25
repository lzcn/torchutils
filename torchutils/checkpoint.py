"""Checkpoint and weight loading utilities."""

import os
from typing import Any

import numpy as np
import torch
from torch import nn

from torchutils.logger import get_logger

from ._internal import set_module

LOGGER = get_logger(__name__)


@set_module("torchutils")
def update_npz(fn, results):
    """Update an ``.npz`` file with new results."""

    if fn is None:
        return
    if os.path.exists(fn):
        pre_results = dict(np.load(fn, allow_pickle=True))
        pre_results.update(results)
        results = pre_results
    np.savez(fn, **results)


@set_module("torchutils")
def weights_init(m):
    """Initialize common layers with Kaiming-normal weights."""

    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


@set_module("torchutils")
def load_pretrained(
    net: nn.Module, path_or_state_dict: Any = None, state_dict=None, weights_only=False, strict=False
) -> nn.Module:
    """Load weights loosely or strictly and log any mismatches."""

    if state_dict is not None:
        path_or_state_dict = state_dict
    assert path_or_state_dict is not None, "path_or_state_dict must be given"

    if isinstance(path_or_state_dict, str):
        LOGGER.info("Loading pre-trained model from %s.", path_or_state_dict)
        state_dict = torch.load(path_or_state_dict, map_location="cpu", weights_only=weights_only)
    else:
        LOGGER.info("Loading pre-trained model from state dict.")
        state_dict = path_or_state_dict

    net_param = net.state_dict()
    unmatched_keys = []
    for name, param in state_dict.items():
        if name in net_param and param.shape != net_param[name].shape:
            unmatched_keys.append(name)
    for name in unmatched_keys:
        state_dict.pop(name)
    missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
    missing_keys = list(set(missing_keys) - set(unmatched_keys))
    LOGGER.info("Missing keys: %s", ", ".join(missing_keys))
    LOGGER.info("Unexpected keys: %s", ", ".join(unexpected_keys))
    LOGGER.info("Unmatched keys: %s", ", ".join(unmatched_keys))
    if strict:
        assert len(missing_keys) == len(unexpected_keys) == len(unmatched_keys) == 0
    return net
