from functools import partial, wraps

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

    The last FC lays is remvoed for all backbone.

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
