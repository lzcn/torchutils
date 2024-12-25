import json
import logging
import operator
import os
from numbers import Number
from typing import IO, Any

import numpy as np
import torch
import yaml
from colorama import Back, Fore, Style
from torch import nn

from .overrides import set_module

LOGGER = logging.getLogger(__name__)

__all__ = [
    "colour",
    "format_display",
    "from_yaml",
    "gather_loss",
    "gather_mean",
    "get_named_class",
    "get_named_function",
    "infer_parallel_device",
    "init_optimizer",
    "load_pretrained",
    "one_hot",
    "to_device",
    "YAMLoader",
]


@set_module("torchutils")
def format_display(opt, num=1, symbol=" "):
    """Convert dictionary to format string.

    Args:
        opt (dict): configuration to be displayed
        num (int): number of indent
    """
    indent = symbol * num
    if isinstance(opt, dict):
        repr_list = ["{}: {}".format(k, format_display(v, num + 1, symbol)) for k, v in opt.items()]
        lsign = "{"
        rsign = "}"
        if sum(map(len, repr_list)) < 10:
            string = lsign + ", ".join(repr_list) + rsign
        else:
            string = lsign + "\n"
            for repr in repr_list:
                string += "{}{},\n".format(indent, repr)
            string += symbol * (num - 1) + rsign
    elif isinstance(opt, list):
        repr_list = [format_display(v, num + 1, symbol) for v in opt]
        lsign = "["
        rsign = "]"
        if sum(map(len, repr_list)) < 10:
            string = lsign + ", ".join(repr_list) + rsign
        else:
            string = lsign + "\n"
            for repr in repr_list:
                string += "{}{},\n".format(indent, repr)
            string += symbol * (num - 1) + rsign
    else:
        string = str(opt)
    return string


def update_npz(fn, results):
    if fn is None:
        return
    if os.path.exists(fn):
        pre_results = dict(np.load(fn, allow_pickle=True))
        pre_results.update(results)
        results = pre_results
    np.savez(fn, **results)


def weights_init(m):
    """
    deprecated
    usage: module.apply(weights_init)
    """
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
    else:
        pass


@set_module("torchutils")
def colour(string, *args, b="", s="", c="green"):
    """Colorize a string.

    Args:
        string (str): string to colorize
        *args (Any): string = string % tuple(args)
        b (str): background color
        s (str): style,
        c (str): foreground color,

    Available formatting:
        See colorma_ for more details

        .. code-block::

            c: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
            b: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
            s: DIM, NORMAL, BRIGHT, RESET_ALL

    .. _colorma: https://pypi.org/project/colorama/

    DEPRECATED:
        The implementation is urgly and not recommended to use.
    """
    if isinstance(string, Number):
        string = str(string)
    string = string % tuple(args)
    prefix = getattr(Fore, c.upper(), "") + getattr(Back, b.upper(), "") + getattr(Style, s.upper(), "")
    suffix = Style.RESET_ALL
    return prefix + string + suffix


@set_module("torchutils")
def get_named_class(module):
    """Get class members in given module."""
    from inspect import isclass

    return {k: v for k, v in module.__dict__.items() if isclass(v) and not k.startswith("_")}


@set_module("torchutils")
def get_named_function(module):
    """Get function members in given module."""
    from inspect import isfunction

    return {k: v for k, v in module.__dict__.items() if isfunction(v) and not k.startswith("_")}


@set_module("torchutils")
def one_hot(index, num):
    """Convert the index tensor to one-hot encoding.

    The returned encoding is on the same device as the index tensor.

    Args:
        index (torch.LongTensor): index tensor
        num (int): length of one-hot encoding

    Returns:
        torch.tensor: one-hot encoding

    Example:

        .. code-block::

            >> x = torch.tensor([2, 1, 1, 2])
            >> one_hot(x, num=3)
            tensor([[0., 0., 1.],
                    [0., 1., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]])

    DEPRECATED:
        use torch.nn.functional.one_hot instead.
    """
    index = index.view(-1, 1)
    one_hot = torch.zeros(index.numel(), num).to(index.device)
    return one_hot.scatter_(1, index, 1.0)


@set_module("torchutils")
def infer_parallel_device(device_ids=None):
    """Decide which device to use for data when given device_ids.

    For :func:`torch.nn.data_parallel`, if multiple GPUs are used, the data only need
    to be saved in CPU. If single GPU is used, then data must be moved to the GPU.

    Args:
        device_ids (list of int, optional): gpu id list

    Outputs:
        - parallel: True if len(device_ids) > 1
        - device: if single-gpu is used then return the gpu device, else return "cpu"
    """
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
def to_device(data, device="cuda"):
    """Move data to given device.

    Args:
        data (Sequence): convert all data to given device.
        device (torch.device, optional): target device.

    """

    error_msg = "data must contains tensors or list of tensors; found {}"
    if isinstance(data, dict):
        return {k: to_device(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return (to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, str):
        return data
    raise TypeError(error_msg.format(type(data)))


@set_module("torchutils")
def gather_loss(loss_dict: dict, loss_weight: dict):
    """Gather overall loss and compute mean of individual losses.

    Args:
        loss_dict (dict): individual loss terms
        loss_weight (dict): weights for each loss, only the loss with valid weight will
            be meaned.

    """
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
    r"""Gather mean value of each tensor.

    Compute the averaged value for each element:
    :math:`\frac{1}{n}\sum_i v_i, v \in\mathbb{R}^n`.

    Args:
        tensors ([dict, list]): list of tensors.
    """
    if isinstance(tensors, dict):
        return {k: v.sum().item() / v.numel() for k, v in tensors.items()}
    elif isinstance(tensors, list):
        return [v.sum().item() / v.numel() for v in tensors]
    else:
        raise TypeError("Expected list or dict, but got {}".format(type(tensors)))


@set_module("torchutils")
def load_pretrained(
    net: nn.Module, path_or_state_dict: Any = None, state_dict=None, weights_only=False, strict=False
) -> nn.Module:
    """Load weights lossly or strictly.

    Load weights that match the the model. Unloaded weighted will be logged.

    Types of unloaded weights:
        - Missing keys: weights not in pretrained state
        - Unexpected keys: weights not in given net
        - Unmatched keys: shape mismatched weights
    Args:
        net (nn.Module): model
        path_or_state_dict (str or dict): path to pre-trained model or state dictionary containing model weights
        state_dict (dict): state dictionary containing model weights. Deprecated. Use path_or_state_dict instead.
        weights_only (bool): whether to load only weights. Default: False
        strict (bool): whether to load weights strictly

    """
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


@set_module("torchutils")
def init_optimizer(net, optim_param):
    """Init Optimizer given OptimParam instance and net."""
    # Optimizer and LR policy class
    grad_class = get_named_class(torch.optim)[optim_param.name]
    lr_class = get_named_class(torch.optim.lr_scheduler)[optim_param.lr_scheduler]
    # Optimizer LR policy configurations
    grad_param = optim_param.grad_param
    lr_param = optim_param.scheduler_param
    # sub-module specific configuration
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
    # get instances
    optimizer = grad_class(param_groups, **grad_param)
    lr_scheduler = lr_class(optimizer, **lr_param)
    return optimizer, lr_scheduler


@set_module("torchutils")
class YAMLoader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor.

    Example:


    """

    def __init__(self, stream: IO) -> None:

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: YAMLoader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip(".")

    with open(filename, "r") as f:
        if extension in ("yaml", "yml"):
            return yaml.load(f, YAMLoader)
        elif extension in ("json",):
            return json.load(f)
        else:
            return "".join(f.readlines())


yaml.add_constructor("!include", construct_include, YAMLoader)


@set_module("torchutils")
def from_yaml(file):
    """Load configuration from YAML file with include constructor.

    To include another file use "!include filename".

    Examples:

        .. code-block::

            # file: bar.yaml
            - 3.6
            - [1, 2, 3]

            # file: foo.yaml
            a: 1
            b:
                - 1.43
                - 543.55
            c: !include bar.yaml

            kwds = from_yaml("foo.yaml")

    Note:
        Under MIT License: Copyright (c) 2018 Josh Bode
    """
    with open(file) as f:
        kwds = yaml.load(f, Loader=YAMLoader)
    return kwds
