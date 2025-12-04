"""Hook utilities for capturing forward features and backward gradients.

This module provides context manager hooks to easily capture intermediate
activations and gradients during model forward and backward passes.

Example::

    import torch
    import torchutils as tu

    model = tu.backbone("resnet50")[0]
    x = torch.randn(1, 3, 224, 224)

    # Capture forward features
    with tu.FeatureHook(model, ["layer2", "layer3"]) as features:
        output = model(x)
    print(features.keys())  # ['layer2', 'layer3']

    # Capture gradients
    with tu.GradHook(model, ["layer2"]) as grads:
        output = model(x)
        loss = output.sum()
        loss.backward()
    print(grads.keys())  # ['layer2']

    # Capture both features and gradients
    with tu.FeatureGradHook(model, ["layer2"]) as (features, grads):
        output = model(x)
        loss = output.sum()
        loss.backward()
"""

from typing import Dict, List, Union

import torch
import torch.nn as nn


class FeatureHook:
    """Capture forward features from specified layers.

    This hook registers forward hooks on specified layers to capture their
    output activations. It works as a context manager and automatically
    cleans up hooks on exit.

    Args:
        model: The neural network model.
        layers: Layer name(s) as string(s). Can be a single layer name
            or a list of layer names (e.g., "layer2" or ["layer2", "layer3"]).

    Returns:
        Dictionary mapping layer names to their output tensors.

    Example::

        model = torchvision.models.resnet50()
        x = torch.randn(1, 3, 224, 224)

        with FeatureHook(model, ["layer2", "layer3.0"]) as features:
            output = model(x)

        print(features["layer2"].shape)  # torch.Size([1, 512, 28, 28])
        print(features["layer3.0"].shape)  # torch.Size([1, 1024, 14, 14])
    """

    def __init__(self, model: nn.Module, layers: Union[str, List[str]]):
        self.model = model
        self.layers = [layers] if isinstance(layers, str) else layers
        self.out: Dict[str, torch.Tensor] = {}
        self.handles = []

    def _hook(self, name: str):
        """Create a forward hook function for the given layer name.

        Args:
            name: Name identifier for the layer.

        Returns:
            Hook function that captures layer output.
        """
        return lambda m, i, o: self.out.setdefault(name, o)

    def __enter__(self):
        """Register forward hooks on specified layers."""
        for name in self.layers:
            module = self.model.get_submodule(name)
            self.handles.append(module.register_forward_hook(self._hook(name)))
        return self.out

    def __exit__(self, exc_type, exc, tb):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()


class GradHook:
    """Capture backward gradients from specified layers.

    This hook registers backward hooks on specified layers to capture their
    output gradients during the backward pass. It works as a context manager
    and automatically cleans up hooks on exit.

    Args:
        model: The neural network model.
        layers: Layer name(s) as string(s). Can be a single layer name
            or a list of layer names (e.g., "layer2" or ["layer2", "layer3"]).

    Returns:
        Dictionary mapping layer names to their output gradient tensors.

    Example::

        model = torchvision.models.resnet50()
        x = torch.randn(1, 3, 224, 224, requires_grad=True)

        with GradHook(model, ["layer2", "layer3"]) as grads:
            output = model(x)
            loss = output.sum()
            loss.backward()

        print(grads["layer2"].shape)  # torch.Size([1, 512, 28, 28])
    """

    def __init__(self, model: nn.Module, layers: Union[str, List[str]]):
        self.model = model
        self.layers = [layers] if isinstance(layers, str) else layers
        self.grads: Dict[str, torch.Tensor] = {}
        self.handles = []

    def _hook(self, name: str):
        """Create a backward hook function for the given layer name.

        Args:
            name: Name identifier for the layer.

        Returns:
            Hook function that captures layer output gradient.
        """
        return lambda m, gi, go: self.grads.setdefault(name, go[0])

    def __enter__(self):
        """Register backward hooks on specified layers."""
        for name in self.layers:
            module = self.model.get_submodule(name)
            self.handles.append(module.register_full_backward_hook(self._hook(name)))
        return self.grads

    def __exit__(self, exc_type, exc, tb):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()


class FeatureGradHook:
    """Capture both forward features and backward gradients.

    This hook combines FeatureHook and GradHook to capture both forward
    activations and backward gradients in a single context. It registers
    both forward and backward hooks on specified layers.

    Args:
        model: The neural network model.
        layers: Layer name(s) as string(s). Can be a single layer name
            or a list of layer names (e.g., "layer2" or ["layer2", "layer3"]).

    Returns:
        Tuple of two dictionaries: (features, gradients), mapping layer
        names to their output tensors and gradient tensors respectively.

    Example::

        model = torchvision.models.resnet50()
        x = torch.randn(1, 3, 224, 224, requires_grad=True)

        with FeatureGradHook(model, ["layer2"]) as (features, grads):
            output = model(x)
            loss = output.sum()
            loss.backward()

        print(features["layer2"].shape)  # torch.Size([1, 512, 28, 28])
        print(grads["layer2"].shape)  # torch.Size([1, 512, 28, 28])
    """

    def __init__(self, model: nn.Module, layers: Union[str, List[str]]):
        self.model = model
        self.layers = [layers] if isinstance(layers, str) else layers
        self.feat: Dict[str, torch.Tensor] = {}
        self.grad: Dict[str, torch.Tensor] = {}
        self.handles = []

    def _feat_hook(self, name: str):
        """Create a forward hook function for the given layer name.

        Args:
            name: Name identifier for the layer.

        Returns:
            Hook function that captures layer output.
        """
        return lambda m, i, o: self.feat.setdefault(name, o)

    def _grad_hook(self, name: str):
        """Create a backward hook function for the given layer name.

        Args:
            name: Name identifier for the layer.

        Returns:
            Hook function that captures layer output gradient.
        """
        return lambda m, gi, go: self.grad.setdefault(name, go[0])

    def __enter__(self):
        """Register forward and backward hooks on specified layers."""
        for name in self.layers:
            module = self.model.get_submodule(name)
            self.handles.append(module.register_forward_hook(self._feat_hook(name)))
            self.handles.append(module.register_full_backward_hook(self._grad_hook(name)))
        return self.feat, self.grad

    def __exit__(self, exc_type, exc, tb):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
