"""torchutils: Essential utilities for PyTorch research and development.

Example::

    import torchutils as tu

    tu.setup_logger(level="INFO", log_file="train.log")
    config = tu.load_config("config.yaml")
    model, dim = tu.backbone("resnet50")
    batch = tu.to(batch, "cuda")
"""

from . import distributed, helpers, hooks, io, logger
from .backbones import backbone
from .config import load_config, save_config
from .distributed import get_rank, rank_zero_only
from .helpers import format_display, get_public_classes, get_public_functions
from .hooks import FeatureGradHook, FeatureHook, GradHook
from .io import ModelSaver, check_exists, load_pretrained, scan_files, scan_folders
from .logger import config as setup_logger, get_logger
from .ops import to

__version__ = "0.0.2"

__all__ = [
    "backbone",
    "load_config",
    "save_config",
    "get_rank",
    "rank_zero_only",
    "ModelSaver",
    "load_pretrained",
    "check_exists",
    "scan_files",
    "scan_folders",
    "setup_logger",
    "get_logger",
    "to",
    "format_display",
    "get_public_classes",
    "get_public_functions",
    "FeatureHook",
    "GradHook",
    "FeatureGradHook",
    "distributed",
    "io",
    "logger",
    "helpers",
    "hooks",
]
