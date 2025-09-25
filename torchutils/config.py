"""Configuration helpers with YAML include support."""

import json
import os
from typing import IO, Any

import yaml

from ._internal import set_module


@set_module("torchutils")
class YAMLoader(yaml.SafeLoader):
    """YAML Loader with ``!include`` constructor."""

    def __init__(self, stream: IO) -> None:
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: YAMLoader, node: yaml.Node) -> Any:
    """Include file referenced at ``node``."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip(".")

    with open(filename) as f:
        if extension in ("yaml", "yml"):
            return yaml.load(f, YAMLoader)
        elif extension in ("json",):
            return json.load(f)
        else:
            return "".join(f.readlines())


yaml.add_constructor("!include", construct_include, YAMLoader)


@set_module("torchutils")
def from_yaml(file):
    """Load configuration from YAML file with ``!include`` support."""

    with open(file) as f:
        kwds = yaml.load(f, Loader=YAMLoader)
    return kwds
