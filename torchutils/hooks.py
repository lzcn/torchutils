"""Capture intermediate features and gradients via context-manager hooks.

Example::

    import torchutils as tu

    with tu.FeatureHook(model, ["layer2", "layer3"]) as features:
        output = model(x)

    with tu.GradHook(model, ["layer1"]) as grads:
        output.sum().backward()
"""

import torch
import torch.nn as nn

__all__ = ["FeatureHook", "GradHook"]


class _HookBase:
    """Register hooks on named submodules and collect outputs into ``self.out``."""

    def __init__(self, model: nn.Module, layers: str | list[str]):
        self.model = model
        self.layers = [layers] if isinstance(layers, str) else layers
        self.out: dict[str, torch.Tensor] = {}
        self.handles = []

    def _register(self, module: nn.Module, name: str):
        raise NotImplementedError

    def __enter__(self) -> dict[str, torch.Tensor]:
        self.out.clear()
        try:
            for name in self.layers:
                module = self.model.get_submodule(name)
                self.handles.append(self._register(module, name))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self.out

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class FeatureHook(_HookBase):
    """Capture forward outputs of the given layers.

    If a layer fires more than once inside the context, the FIRST output
    is kept. Hooks are removed on exit even if registration failed midway
    (an invalid layer name raises AttributeError after cleanup).

    Example::

        with FeatureHook(model, ["layer2", "layer3"]) as features:
            output = model(x)
        print(features["layer2"].shape)
    """

    def _register(self, module: nn.Module, name: str):
        def hook(m, i, o):
            self.out.setdefault(name, o)

        return module.register_forward_hook(hook)


class GradHook(_HookBase):
    """Capture gradients of the given layers' outputs after backward().

    Stores the gradient w.r.t. the module's FIRST output tensor. If a
    layer is reached more than once inside the context, the first
    captured gradient is kept.

    Example::

        with GradHook(model, ["layer2"]) as grads:
            loss = model(x).sum()
            loss.backward()
        print(grads["layer2"].shape)
    """

    def _register(self, module: nn.Module, name: str):
        def hook(m, gi, go):
            self.out.setdefault(name, go[0])

        return module.register_full_backward_hook(hook)
