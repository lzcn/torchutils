"""torchutils: small, DDP-aware utilities for PyTorch training.

Import everything from the top level::

    import torchutils as tu

All functions work in plain single-process scripts; under distributed
training the file-writing / logging entry points act on rank 0 only.
Full guide: docs/usage.md in the repository.

API overview
------------

Logging::

    tu.setup_logger(level="INFO", stream_level=None, file_level=None,
                    log_file=None, file_mode="a", format_string=None,
                    date_format=None)

Configures the root logger with a console handler and an optional file
handler. Only rank 0 emits records under distributed training. Safe to
call repeatedly (handlers are reset each call). Raises ValueError on an
unknown level string.

Device transfer::

    tu.to(data, device="cuda", non_blocking=True) -> data

Recursively moves tensors through nested dicts / lists / tuples.
Namedtuples keep their type; non-tensor leaves pass through unchanged.

Checkpoints::

    saver = tu.ModelSaver(dirname, filename_prefix=None, n_saved=5,
                          save_latest=False, save_best=True, mode="max")
    saver.save(model_or_state_dict, score, epoch=None)

Keeps the best ``n_saved`` scored checkpoints (filenames
``{prefix}_{score:.4f}_epoch_{epoch}.pt``) plus optional
``{prefix}_latest.pt`` and ``{prefix}_best.pt`` copies. Writes are
atomic (temp file + os.replace) and rank-0 only. Re-saving an identical
(score, epoch) overwrites that file in place. After saving:
``saver.best_checkpoint`` holds the best file's path (or None), and
``saver.history`` the kept ``(score, path)`` entries, worst first.
Raises ValueError on invalid ``mode`` or ``n_saved < 1``.

Weight loading::

    tu.load_pretrained(net, path_or_state_dict,
                       weights_only=True, strict=False) -> net

Loads a checkpoint file or a state dict into ``net``, skipping missing
and shape-mismatched keys. With ``strict=True`` it raises RuntimeError
on any missing / unexpected / shape-mismatched key instead. Dicts that
contain a ``"state_dict"`` key (full training checkpoints) are unwrapped
automatically. Files are loaded to CPU first.

Hooks::

    with tu.FeatureHook(model, layers) as features: ...
    with tu.GradHook(model, layers) as grads: ...

``layers`` is one module name or a list of names, resolved with
``nn.Module.get_submodule`` (e.g. "layer2", "encoder.blocks.3").
Captured tensors are collected in the dict yielded by the context
manager; if a module fires more than once, the FIRST output is kept.
GradHook stores the gradient of the module's first output tensor.
Hooks are always removed on exit, even when a name is invalid (in that
case AttributeError is raised after cleanup).

Misc::

    @tu.rank_zero_only
    def notify(): ...

Runs the function only on global rank 0 (no-op -> returns None on
other ranks).

    tu.scan_files(path="./", suffix=(), recursive=False, relpath=False)

Lists files under ``path``, skipping hidden (dot-prefixed) entries.
``suffix`` is a string or tuple; the leading dot is optional, so
"jpg" and ".jpg" are equivalent. An empty suffix means no filtering.
Returns str paths (relative to ``path`` when ``relpath=True``); the
order follows the filesystem and is not sorted.
"""

from .checkpoint import ModelSaver, load_pretrained
from .distributed import rank_zero_only
from .filesystem import scan_files
from .hooks import FeatureHook, GradHook
from .logger import setup_logger
from .ops import to

__version__ = "1.1.0"

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
