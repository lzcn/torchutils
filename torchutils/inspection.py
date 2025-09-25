"""Reflection helpers for discovering module members."""

from ._internal import set_module


@set_module("torchutils")
def get_named_class(module):
    """Return public classes defined in ``module`` as a ``dict``."""

    from inspect import isclass

    return {k: v for k, v in module.__dict__.items() if isclass(v) and not k.startswith("_")}


@set_module("torchutils")
def get_named_function(module):
    """Return public functions defined in ``module`` as a ``dict``."""

    from inspect import isfunction

    return {k: v for k, v in module.__dict__.items() if isfunction(v) and not k.startswith("_")}
