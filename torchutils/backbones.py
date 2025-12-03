"""Backbone model registry for common computer vision architectures.

Backbones have their final classification layer replaced with an identity mapping,
making them suitable for feature extraction.

Example::

    import torchutils as tu
    
    model, dim = tu.backbone("resnet50")
"""

from functools import partial
from typing import Callable, Dict, Tuple, Union

import torch.nn as nn
from torchvision import models

# Global registry mapping backbone names to factory functions
_BACKBONES: Dict[str, Callable] = {}


def register(name: str, model_fn: Callable, feature_dim: int) -> None:
    """Register a backbone model.

    Args:
        name: Unique identifier for the backbone.
        model_fn: Factory function that creates the model.
        feature_dim: Output feature dimension of the backbone.

    Example::

        def create_custom_backbone(weights=None):
            model = MyModel(pretrained=(weights == "DEFAULT"))
            model.fc = nn.Identity()
            return model, 512

        register("custom", create_custom_backbone, 512)
    """
    if name in _BACKBONES:
        raise ValueError(f"Backbone '{name}' is already registered.")
    _BACKBONES[name] = partial(_create_backbone, model_fn, feature_dim)


def backbone(name: str, weights: Union[str, None] = "DEFAULT", **kwargs) -> Tuple[nn.Module, int]:
    """Retrieve a backbone model by name.

    Args:
        name: Name of the backbone (e.g., "resnet50", "efficientnet_b0").
        weights: Pretrained weights to load. Defaults to "DEFAULT". Use None for random initialization.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        Tuple of (model, feature_dim) where model is the backbone with fc replaced
        by Identity, and feature_dim is the output feature dimension.

    Raises:
        ValueError: If the backbone name is not registered.

    Example::

        >>> model, dim = backbone("resnet18")
        >>> x = torch.randn(1, 3, 224, 224)
        >>> features = model(x)
        >>> print(features.shape)  # (1, 512)
    """
    if name not in _BACKBONES:
        available = ", ".join(sorted(_BACKBONES.keys()))
        raise ValueError(f"Unknown backbone '{name}'. Available: {available}")
    return _BACKBONES[name](weights=weights, **kwargs)


def _create_backbone(
    model_fn: Callable, feature_dim: int, weights: Union[str, None] = "DEFAULT", **kwargs
) -> Tuple[nn.Module, int]:
    """Internal helper to create a backbone with identity fc layer.

    Args:
        model_fn: Constructor for torchvision model.
        feature_dim: Output feature size.
        weights: Weights to load.
        **kwargs: Additional model arguments.

    Returns:
        Tuple of (model, feature_dim).
    """
    model = model_fn(weights=weights, **kwargs)
    model.fc = nn.Identity()
    return model, feature_dim


# Registry of common torchvision backbones
_TORCHVISION_BACKBONES = [
    ("resnet18", models.resnet18, 512),
    ("resnet34", models.resnet34, 512),
    ("resnet50", models.resnet50, 2048),
    ("resnet101", models.resnet101, 2048),
    ("resnet152", models.resnet152, 2048),
    ("mobilenet_v2", models.mobilenet_v2, 1280),
    ("mobilenet_v3_large", models.mobilenet_v3_large, 1280),
    ("mobilenet_v3_small", models.mobilenet_v3_small, 576),
    ("efficientnet_b0", models.efficientnet_b0, 1280),
    ("efficientnet_b1", models.efficientnet_b1, 1280),
    ("efficientnet_b2", models.efficientnet_b2, 1408),
    ("efficientnet_b3", models.efficientnet_b3, 1536),
    ("efficientnet_b4", models.efficientnet_b4, 1792),
    ("efficientnet_b5", models.efficientnet_b5, 2048),
    ("efficientnet_b6", models.efficientnet_b6, 2304),
    ("efficientnet_b7", models.efficientnet_b7, 2560),
]

# Auto-register all torchvision backbones
for name, model_fn, feature_dim in _TORCHVISION_BACKBONES:
    _BACKBONES[name] = partial(_create_backbone, model_fn, feature_dim)
