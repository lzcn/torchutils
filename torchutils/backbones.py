import operator
from functools import partial, wraps
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models

from torchutils.overrides import set_module

_BACKBONES: Dict[str, Callable] = {}


def register_backbone(func: Callable = None, *, name: str = None) -> Callable:
    if func is None:
        return partial(register_backbone, name=name)

    model_name = name if name else func.__name__
    assert model_name not in _BACKBONES, f"{model_name} is already registered."
    _BACKBONES[model_name] = func

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@set_module("torchutils")
def backbone(name: str, weights: str = "DEFAULT", **kwargs) -> Tuple[nn.Module, int]:
    """Get backbone by name.

    The last FC layer is removed for all backbones.

    Args:
        name (str): the name of backbone
        weights (str, optional): the weights to load. Defaults to "DEFAULT".

    Raises:
        ValueError: if the backbone is unknown

    Returns:
        Tuple[nn.Module, int]: an instance of the backbone, number of features
    """
    if name in _BACKBONES:
        return _BACKBONES[name](weights=weights, **kwargs)
    else:
        raise ValueError(f"Unknown backbone {name}")


def create_backbone(
    model_fn: Callable, num_features: int, weights: str = "DEFAULT", replace_bn: bool = False, **kwargs
) -> Tuple[nn.Module, int]:
    backbone = model_fn(weights=weights, **kwargs)
    if replace_bn:
        backbone = _replace_bn(backbone)
    backbone.fc = nn.Identity()
    return backbone, num_features


def register_torchvision_backbones():
    torchvision_models = [
        ("alexnet", 4096),
        ("vgg11", 4096),
        ("vgg16", 4096),
        ("vgg19", 4096),
        ("resnet18", 512),
        ("resnet34", 512),
        ("resnet50", 2048),
        ("resnet101", 2048),
        ("resnet152", 2048),
        ("mobilenet_v2", 1280),
        ("mobilenet_v3_large", 1280),
        ("mobilenet_v3_small", 576),
        ("efficientnet_b0", 1280),
        ("efficientnet_b1", 1280),
        ("efficientnet_b2", 1408),
        ("efficientnet_b3", 1536),
        ("efficientnet_b4", 1792),
        ("efficientnet_b5", 2048),
        ("efficientnet_b6", 2304),
        ("efficientnet_b7", 2560),
    ]

    for model_name, num_features in torchvision_models:
        register_backbone(partial(create_backbone, getattr(models, model_name), num_features), name=model_name)
        register_backbone(
            partial(create_backbone, getattr(models, model_name), num_features, replace_bn=True),
            name=f"{model_name}_affine",
        )


class AffineBatchNorm2d(nn.Module):
    """BatchNorm2d without tracking."""

    def __init__(self, num_features: int):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.num_features = num_features
        self.eps = 1e-5

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}"

    def _t(self, w: torch.Tensor) -> torch.Tensor:
        return w.view(self.num_features, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = (x - self._t(self.running_mean)) / torch.sqrt(self._t(self.running_var) + self.eps)
        return self._t(self.weight) * h + self._t(self.bias)


def _replace_bn(resnet: nn.Module) -> nn.Module:
    state_dict = resnet.state_dict()
    to_replace = [
        (name, AffineBatchNorm2d(module.num_features)) for name, module in resnet.named_modules() if "bn" in name
    ]
    for name, value in to_replace:
        name_parts = name.split(".")
        parent_module = operator.attrgetter(".".join(name_parts[:-1]))(resnet) if len(name_parts) > 1 else resnet
        setattr(parent_module, name_parts[-1], value)
    resnet.load_state_dict(state_dict)
    return resnet


register_torchvision_backbones()
