from functools import partial, wraps
from typing import Callable, Dict, Tuple, Union

import torch.nn as nn
from torchvision import models

# Global registry for all backbone models
_BACKBONES: Dict[str, Callable] = {}


def register_backbone(func: Callable = None, *, name: str = None) -> Callable:
    """Register a custom backbone model.

    Args:
        func (Callable): The function that returns a model.
        name (str, optional): The name to register under. Defaults to the function's name.

    Returns:
        Callable: A wrapped function that returns the model.
    """
    if func is None:
        return partial(register_backbone, name=name)

    model_name = name if name else func.__name__
    assert model_name not in _BACKBONES, f"{model_name} is already registered."
    _BACKBONES[model_name] = func

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def backbone(name: str, weights: Union[str, None] = "DEFAULT", **kwargs) -> Tuple[nn.Module, int]:
    """Retrieve a backbone model by name.

    Args:
        name (str): Name of the backbone.
        weights (Union[str, None], optional): Pretrained weights to load. Defaults to "DEFAULT".

    Raises:
        ValueError: If the backbone name is not registered.

    Returns:
        Tuple[nn.Module, int]: Model instance and output feature dimension.

    Example:
        >>> model, dim = backbone("resnet18")
        >>> x = torch.randn(1, 3, 224, 224)
        >>> out = model(x)
        >>> print(out.shape)  # (1, 512)
    """
    if name in _BACKBONES:
        return _BACKBONES[name](weights=weights, **kwargs)
    else:
        raise ValueError(f"Unknown backbone '{name}'. Registered: {list(_BACKBONES.keys())}")


def create_backbone(
    model_fn: Callable, num_features: int, weights: Union[str, None] = "DEFAULT", **kwargs
) -> Tuple[nn.Module, int]:
    """Create a backbone model with optional batch norm replacement.

    Args:
        model_fn (Callable): Constructor for torchvision model.
        num_features (int): Output feature size.
        weights (Union[str, None], optional): Weights to load. Defaults to "DEFAULT".

    Returns:
        Tuple[nn.Module, int]: Modified model and output feature dimension.
    """
    backbone = model_fn(weights=weights, **kwargs)
    backbone.fc = nn.Identity()
    return backbone, num_features


def register_torchvision_backbones():
    torchvision_models = [
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
        model_fn = getattr(models, model_name)
        register_backbone(partial(create_backbone, model_fn, num_features), name=model_name)


# Register all torchvision backbones on import
register_torchvision_backbones()
