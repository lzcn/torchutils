"""Training helpers for losses, devices, and optimizers."""

import operator
from typing import Any, Mapping

import torch

from ._internal import set_module
from .inspection import get_named_class


@set_module("torchutils")
def infer_parallel_device(device_ids=None):
    """Decide which device to use for data in ``torch.nn.data_parallel`` setups."""

    device_ids = [] if device_ids is None else device_ids
    if len(device_ids) == 0:
        parallel = False
        device = torch.device("cpu")
        return parallel, device
    elif len(device_ids) == 1:
        parallel = False
        device = torch.device(device_ids[0])
    else:
        parallel = True
        device = torch.device("cpu")
    return parallel, device


@set_module("torchutils")
def gather_loss(loss_dict: Mapping[str, torch.Tensor], loss_weight: Mapping[str, Any]):
    """Gather overall loss and compute mean of individual losses."""

    loss = 0.0
    scale_dict = {}
    for name, value in loss_dict.items():
        value = loss_dict[name].mean()
        weight = loss_weight.get(name, None)
        if weight:
            loss += value * weight
        scale_dict[name] = value.item()
    return scale_dict, loss


@set_module("torchutils")
def gather_mean(tensors):
    r"""Gather mean value of each tensor."""

    if isinstance(tensors, dict):
        return {k: v.sum().item() / v.numel() for k, v in tensors.items()}
    elif isinstance(tensors, list):
        return [v.sum().item() / v.numel() for v in tensors]
    else:
        raise TypeError(f"Expected list or dict, but got {type(tensors)}")


@set_module("torchutils")
def init_optimizer(net, optim_param):
    """Initialise an optimizer and scheduler using ``OptimParam`` metadata."""

    grad_class = get_named_class(torch.optim)[optim_param.name]
    lr_class = get_named_class(torch.optim.lr_scheduler)[optim_param.lr_scheduler]
    grad_param = optim_param.grad_param
    lr_param = optim_param.scheduler_param
    named_groups = optim_param.groups
    param_groups = []
    if named_groups:
        param_groups = []
        for name, groups in named_groups.items():
            sub_module = operator.attrgetter(name)(net)
            param_group = dict(params=sub_module.parameters(), **groups)
            param_groups.append(param_group)
    else:
        param_group = net.parameters()
    optimizer = grad_class(param_groups, **grad_param)
    lr_scheduler = lr_class(optimizer, **lr_param)
    return optimizer, lr_scheduler
