from torchutils.backbones import backbone
from torchutils.misc import *  # noqa F401
# from torchutils.singleton import singleton

# TODO: remove param module

from . import (
    data,
    dist,
    draw,
    factory,
    files,
    ignite,
    init,
    io,
    layers,
    logger,
    loss,
    meter,
    metrics,
    ops,
    ot,
    param,
    plot,
    singleton,
)

__all__ = [
    "backbone",
    "data",
    "dist",
    "draw",
    "factory",
    "files",
    "ignite",
    "init",
    "io",
    "layers",
    "logger",
    "loss",
    "meter",
    "metrics",
    "ops",
    "ot",
    "param",
    "plot",
    "singleton"
]
__version__ = "0.0.1-dev211030"
