import operator
from functools import partial, wraps

import torch
import torch.nn as nn
from torchvision import models

from .overrides import set_module

_BACKBONES = {}


def register_backbone(func=None, *, name=None):
    if func is None:
        return partial(register_backbone, name=name)

    model_name = name if name else func.__name__
    assert model_name not in _BACKBONES, f"{model_name} is already registered."
    _BACKBONES[model_name] = func

    @wraps(func)
    def wrapper():
        return func

    return wrapper


@set_module("torchutils")
def backbone(name: str, pretrained=True, **kwargs):
    """Get backbone by name.

    The last FC lays is removed for all backbone.

    Args:
        name (str): the name of backbone
        pretrained (bool, optional): if pretrained. Defaults to True.

    Raises:
        ValueError: [description]

    Returns:
        List[nn.Module, int]: an instance of the backbone, number of features
    """
    if name in _BACKBONES:
        return _BACKBONES[name](pretrained=pretrained, **kwargs)
    else:
        raise ValueError("Unknown backbone {:s}".format(name))


@register_backbone
def alexnet(pretrained=True, **kwargs):
    num_features = 4096
    backbone = models.alexnet(pretrained, **kwargs)
    backbone.classifier[-1] = nn.Identity()
    return backbone, num_features


@register_backbone
def inception_v3(pretrained=True, **kwargs):
    num_features = 2048
    backbone = models.inception_v3(pretrained, aux_logits=False, **kwargs)
    backbone.fc = nn.Identity()
    return backbone, num_features


@register_backbone
def resnet18(pretrained=True, **kwargs):
    num_features = 512
    backbone = models.resnet18(pretrained, **kwargs)
    backbone.fc = nn.Identity()
    return backbone, num_features


@register_backbone
def resnet34(pretrained=True, **kwargs):
    num_features = 512
    backbone = models.resnet34(pretrained, **kwargs)
    backbone.fc = nn.Identity()
    return backbone, num_features


@register_backbone
def resnet50(pretrained=True, **kwargs):
    num_features = 2048
    backbone = models.resnet50(pretrained, **kwargs)
    backbone.fc = nn.Identity()
    return backbone, num_features


@register_backbone
def resnet101(pretrained=True, **kwargs):
    num_features = 2048
    backbone = models.resnet101(pretrained, **kwargs)
    backbone.fc = nn.Identity()
    return backbone, num_features


@register_backbone
def resnet152(pretrained=True, **kwargs):
    num_features = 2048
    backbone = models.resnet152(pretrained, **kwargs)
    backbone.fc = nn.Identity()
    return backbone, num_features


class AffineBatchNorm2d(nn.Module):
    """BatchNorm2d without tracking."""

    def __init__(self, num_features):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.num_features = num_features
        self.eps = 1e-5

    def extra_repr(self):
        return "{num_features}, eps={eps}".format(**self.__dict__)

    def _t(self, w):
        return w.view(self.num_features, 1, 1)

    def forward(self, x):
        h = (x - self._t(self.running_mean)) / torch.sqrt(self._t(self.running_var) + self.eps)
        return self._t(self.weight) * h + self._t(self.bias)


def _replace_bn(resnet: nn.Module):
    state_dict = resnet.state_dict()
    to_replace = []
    for name, module in resnet.named_modules():
        if "bn" in name:
            to_replace.append((name, AffineBatchNorm2d(module.num_features)))
    for name, value in to_replace:
        name = name.split(".")
        if len(name) == 1:
            setattr(resnet, name[0], value)
        else:
            name1 = ".".join(name[:-1])
            name2 = name[-1]
            setattr(operator.attrgetter(name1)(resnet), name2, value)
    resnet.load_state_dict(state_dict)
    return resnet


@register_backbone
def resnet18_affine(pretrained=True, **kwargs):
    num_features = 512
    backbone = models.resnet18(pretrained, **kwargs)
    backbone = _replace_bn(backbone)
    backbone.fc = nn.Identity()
    return backbone, num_features


@register_backbone
def resnet34_affine(pretrained=True, **kwargs):
    num_features = 512
    backbone = models.resnet34(pretrained, **kwargs)
    backbone = _replace_bn(backbone)
    backbone.fc = nn.Identity()
    return backbone, num_features


@register_backbone
def resnet50_affine(pretrained=True, **kwargs):
    num_features = 2048
    backbone = models.resnet50(pretrained, **kwargs)
    backbone = _replace_bn(backbone)
    backbone.fc = nn.Identity()
    return backbone, num_features


@register_backbone
def resnet101_affine(pretrained=True, **kwargs):
    num_features = 2048
    backbone = models.resnet101(pretrained, **kwargs)
    backbone = _replace_bn(backbone)
    backbone.fc = nn.Identity()
    return backbone, num_features


@register_backbone
def resnet152_affine(pretrained=True, **kwargs):
    num_features = 2048
    backbone = models.resnet152(pretrained, **kwargs)
    backbone = _replace_bn(backbone)
    backbone.fc = nn.Identity()
    return backbone, num_features
