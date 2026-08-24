import os
from pathlib import Path
import stat
import tempfile

import torch
from torch import nn

from .distributed import _rank

__all__ = ["ModelSaver", "load_pretrained"]


def load_pretrained(
    net: nn.Module,
    path_or_state_dict: str | Path | dict,
    weights_only=True,
    strict=False,
) -> nn.Module:
    """Load weights into a network, skipping missing or shape-mismatched keys.

    Args:
        net: The neural network module to load weights into.
        path_or_state_dict: Checkpoint file path or a state dict.
        weights_only: If True, only weights will be unpickled (recommended for security).
        strict: If True, raises an error when anything is missing, unexpected,
            or shape-mismatched.

    Returns:
        The network with loaded weights.
    """
    if isinstance(path_or_state_dict, (str, Path)):
        state_dict = torch.load(
            path_or_state_dict, map_location="cpu", weights_only=weights_only
        )
    else:
        state_dict = path_or_state_dict

    net_param = net.state_dict()
    filtered = {}
    unmatched_keys = []
    for name, param in state_dict.items():
        if name in net_param and isinstance(param, torch.Tensor) and param.shape != net_param[name].shape:
            unmatched_keys.append(name)
        else:
            filtered[name] = param

    missing_keys, unexpected_keys = net.load_state_dict(filtered, strict=False)
    if strict and (missing_keys or unexpected_keys or unmatched_keys):
        raise RuntimeError(
            f"Strict loading failed: {len(missing_keys)} missing, "
            f"{len(unexpected_keys)} unexpected, {len(unmatched_keys)} unmatched keys."
        )
    return net


class ModelSaver:
    """Keep the best and/or most recent checkpoints on disk.

    Filenames look like ``{prefix}_{score:.4f}_epoch_{epoch}.pt``, plus a
    ``{prefix}_best.pt`` / ``{prefix}_latest.pt`` copy when enabled. Writes are
    atomic (no corrupted half-written files) and only rank 0 writes under
    distributed training.

    Args:
        dirname: Directory where checkpoints are saved (created if missing).
        filename_prefix: Optional prefix for checkpoint filenames.
        n_saved: Number of scored checkpoints to keep.
        save_latest: If True, also keep a rolling ``{prefix}_latest.pt``.
        save_best: If True, also keep a copy of the best checkpoint.
        mode: "max" or "min" - whether higher or lower scores are better.

    Example::

        saver = ModelSaver("checkpoints", n_saved=3)
        saver.save(model, score=0.95, epoch=10)
        load_pretrained(model, saver.best_checkpoint)
    """

    def __init__(
        self,
        dirname: str,
        filename_prefix: str = None,
        n_saved: int = 5,
        save_latest: bool = False,
        save_best: bool = True,
        mode: str = "max",
    ):
        if mode not in ("max", "min"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'max' or 'min'.")

        self.dirname = dirname
        self.filename_prefix = filename_prefix
        self.n_saved = n_saved
        self.save_latest = save_latest
        self.save_best = save_best
        self.mode = mode
        self.best_checkpoint: str = None
        # history of (score, filename), sorted worst -> best
        init_score = float("inf") if mode == "min" else -float("inf")
        self.history = [(init_score, None)] * n_saved

        os.makedirs(dirname, exist_ok=True)

    def _path(self, name: str) -> str:
        prefix = f"{self.filename_prefix}_" if self.filename_prefix else ""
        return os.path.join(self.dirname, f"{prefix}{name}.pt")

    def _scored_path(self, score: float, epoch: int = None) -> str:
        parts = [f"{self.filename_prefix}_"] if self.filename_prefix else []
        parts.append(f"{score:.4f}")
        if epoch is not None:
            parts.append(f"_epoch_{epoch}")
        return os.path.join(self.dirname, "".join(parts) + ".pt")

    def _save(self, state_dict: dict, path: str) -> None:
        tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.dirname)
        try:
            torch.save(state_dict, tmp.file)
        except BaseException:
            tmp.close()
            os.remove(tmp.name)
            raise
        tmp.close()
        os.replace(tmp.name, path)
        # make group/others readable (shared cluster dirs)
        os.chmod(path, os.stat(path).st_mode | stat.S_IRGRP | stat.S_IROTH)

    def _is_worst(self, score) -> bool:
        # history[0] is the worst kept entry
        return (
            score < self.history[0][0]
            if self.mode == "max"
            else score > self.history[0][0]
        )

    def _is_best(self, score) -> bool:
        return (
            score >= self.history[-1][0]
            if self.mode == "max"
            else score <= self.history[-1][0]
        )

    def save(self, model: nn.Module, score: float, epoch: int = None) -> None:
        """Save a checkpoint if it ranks among the best ``n_saved`` scores."""
        if _rank() != 0:
            return

        state_dict = model.state_dict()
        if self.save_latest:
            self._save(state_dict, self._path("latest"))

        if self._is_worst(score):
            return

        new_best = self._is_best(score)
        # write the new checkpoint BEFORE removing the old one (crash-safe)
        path = self._scored_path(score, epoch)
        self._save(state_dict, path)

        worst_file = self.history[0][1]
        if worst_file and os.path.exists(worst_file):
            os.remove(worst_file)

        self.history[0] = (score, path)
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "min"))

        if self.save_best and new_best:
            best_path = self._path("best")
            self._save(state_dict, best_path)
            self.best_checkpoint = best_path
