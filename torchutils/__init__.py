"""torchutils: Essential utilities for PyTorch research and development.

Example::

    import torchutils as tu

    tu.setup_logger(level="INFO", log_file="train.log")
    batch = tu.to(batch, "cuda")
    saver = tu.ModelSaver("checkpoints", n_saved=3)
"""

from .checkpoint import ModelSaver, load_pretrained
from .distributed import rank_zero_only
from .filesystem import scan_files
from .hooks import FeatureHook, GradHook
from .logger import setup_logger
from .ops import to

__version__ = "0.1.0"

__all__ = [
    "FeatureHook",
    "GradHook",
    "ModelSaver",
    "load_pretrained",
    "rank_zero_only",
    "scan_files",
    "setup_logger",
    "to",
]
