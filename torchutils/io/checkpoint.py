import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Union

import numpy as np
import torch
from torch import nn
import torch.distributed as dist

try:
    from ..logger import get_logger
except ImportError:
    from logging import getLogger as get_logger

PathLike = Union[str, Path]

logger = get_logger(__name__)


def load_pretrained(net: nn.Module, path_or_state_dict: Any = None, weights_only=True, strict=False) -> nn.Module:
    """Load weights loosely or strictly and log any mismatches.

    Args:
        net: The neural network module to load weights into.
        path_or_state_dict: Either a file path (str) to load checkpoint from, or a state dict (dict).
        weights_only: If True, only weights will be unpickled (recommended for security).
        strict: If True, raises an error when there are missing, unexpected, or unmatched keys.

    Returns:
        The network with loaded weights.

    Note:
        * ``missing_keys`` is a list of str containing any keys that are expected
          by this module but missing from the provided ``state_dict``.
        * ``unexpected_keys`` is a list of str containing the keys that are not
          expected by this module but present in the provided ``state_dict``.
        * ``unmatched_keys`` is a list of str containing the keys with shape mismatches
          between the module and the provided ``state_dict`` (these are skipped).
    """

    assert path_or_state_dict is not None, "path_or_state_dict must be given"

    if isinstance(path_or_state_dict, str):
        logger.info("Loading pre-trained model from %s.", path_or_state_dict)
        state_dict = torch.load(path_or_state_dict, map_location="cpu", weights_only=weights_only)
    else:
        logger.info("Loading pre-trained model from state dict.")
        state_dict = path_or_state_dict

    net_param = net.state_dict()
    unmatched_keys = []
    for name, param in state_dict.items():
        if name in net_param and param.shape != net_param[name].shape:
            unmatched_keys.append(name)
    for name in unmatched_keys:
        state_dict.pop(name)
    missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
    missing_keys = list(set(missing_keys) - set(unmatched_keys))
    if missing_keys:
        logger.info("Missing keys: %s", ", ".join(missing_keys))
    if unexpected_keys:
        logger.info("Unexpected keys: %s", ", ".join(unexpected_keys))
    if unmatched_keys:
        logger.info("Unmatched keys: %s", ", ".join(unmatched_keys))
    if strict:
        assert len(missing_keys) == len(unexpected_keys) == len(unmatched_keys) == 0
    return net


class ModelSaver:
    """Handler that saves model checkpoints to disk.

    Filename format: {prefix}[_{score_name}]_{score:.4f}[_{epoch}].pt
    Optional: {prefix}_latest.pt, {prefix}_best.pt

    Args:
        dirname: Directory path where checkpoints will be saved.
        filename_prefix: Prefix for checkpoint filenames.
        score_name: Name for the score metric.
        n_saved: Number of checkpoints to keep.
        save_latest: If True, always save latest checkpoint.
        save_best: If True, save a copy of the best checkpoint.
        mode: "max" or "min" for score comparison.
        atomic: If True, use atomic writes to prevent corruption.
        create_dir: If True, create directory if it doesn't exist.
        require_empty: If True, raise error if directory is not empty.

    Example::

        saver = ModelSaver("checkpoints", n_saved=5, save_best=True)
        for epoch in range(max_epochs):
            score = evaluate(model)
            saver.save(model, score, epoch)
    """

    def __init__(
        self,
        dirname: str,
        filename_prefix: str = None,
        score_name: str = None,
        n_saved: int = 5,
        save_latest: bool = False,
        save_best: bool = False,
        mode="max",
        atomic: bool = True,
        create_dir: bool = True,
        require_empty: bool = True,
    ):
        super().__init__()
        score = np.inf if mode == "min" else -np.inf
        self.dirname = dirname
        self.filename_prefix = filename_prefix
        self.score_name = score_name
        self.n_saved = n_saved
        self.save_latest = save_latest
        self.save_best = save_best
        self.mode = mode
        self.history = [(score, False) for _ in range(n_saved)]
        self.best_checkpoint: str = None
        self.atomic = atomic

        if create_dir and not os.path.exists(dirname):
            os.makedirs(dirname)
        if require_empty and os.listdir(dirname):
            raise ValueError(f"Directory {dirname} is not empty")

    def filename(self, score: float = None, epoch: int = None, latest=False, best=False) -> str:
        prefix = f"{self.filename_prefix}_" if self.filename_prefix else ""
        score_name = f"{self.score_name}_" if self.score_name else ""
        if latest:
            return f"{self.dirname}/{prefix}latest.pt"
        if best:
            return f"{self.dirname}/{prefix}best.pt"
        if score is None and epoch is not None:
            filename = f"{self.dirname}/{prefix}epoch_{epoch}.pt"
        elif score is not None and epoch is None:
            filename = f"{self.dirname}/{prefix}{score_name}{score:.4f}.pt"
        else:
            filename = f"{self.dirname}/{prefix}{score_name}{score:.4f}_epoch_{epoch}.pt"
        return filename

    def _is_worst(self, score):
        # assume that self.history is sorted with first being the worst
        return (score < self.history[0][0]) if self.mode == "max" else (score > self.history[0][0])

    def _is_best(self, score):
        # assume that self.history is sorted with last item being the best
        return (score >= self.history[-1][0]) if self.mode == "max" else (score <= self.history[-1][0])

    def _sort_history(self):
        # sort the history from worst to best
        if self.mode == "max":
            self.history.sort(key=lambda x: x[0])
        else:
            self.history.sort(key=lambda x: x[0], reverse=True)

    def _save_func(self, checkpoint: dict, path: Path, func: callable) -> None:
        if not self.atomic:
            func(checkpoint, path)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.dirname)
            tmp_file = tmp.file
            tmp_name = tmp.name
            try:
                func(checkpoint, tmp_file)
            except BaseException:
                tmp.close()
                os.remove(tmp_name)
                raise
            else:
                tmp.close()
                os.replace(tmp.name, path)
                # append group/others read mode
                os.chmod(path, os.stat(path).st_mode | stat.S_IRGRP | stat.S_IROTH)

    def save(self, model: nn.Module, score, epoch=None):
        """Save model checkpoint.

        The format of the filename is ``{filename_prefix}[_{score_name}]_{score:.4f}[_{epoch}].pt``
        where [_{score_name}] and [_{epoch}] are optional.

        Args:
            model (nn.Module): model to save
            score (float, Optional): score
            epoch (Number, Optional): current epoch
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        if self._is_worst(score) and not self.save_latest:
            return
        state_dict = model.state_dict()
        if self.save_latest:
            self._save_func(state_dict, self.filename(latest=True), torch.save)
        if self.save_best and self._is_best(score):
            self._save_func(state_dict, self.filename(best=True), torch.save)
        filename = self.filename(score, epoch)
        if self._is_worst(score):
            pass
        else:
            # replace the worst model
            if self.history[0][1] and os.path.exists(self.history[0][1]):
                os.remove(self.history[0][1])
            self._save_func(state_dict, filename, torch.save)
            self.history[0] = (score, filename)
        self._sort_history()
        self.best_checkpoint = self.history[-1][-1]
