"""Backward compatibility imports for legacy ``torchutils.misc`` usage."""

from .checkpoint import load_pretrained, update_npz, weights_init
from .config import YAMLoader, construct_include, from_yaml
from .formatting import format_display
from .inspection import get_named_class, get_named_function
from .tensor import one_hot
from .training import gather_loss, gather_mean, infer_parallel_device, init_optimizer
from torchutils.ops import to

__all__ = [
    "format_display",
    "update_npz",
    "weights_init",
    "get_named_class",
    "get_named_function",
    "one_hot",
    "infer_parallel_device",
    "to",
    "gather_loss",
    "gather_mean",
    "load_pretrained",
    "init_optimizer",
    "YAMLoader",
    "construct_include",
    "from_yaml",
]
