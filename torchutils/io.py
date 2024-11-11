import csv
import json
import os
import numpy as np
from torch import nn
import torch
import tempfile
import stat
from pathlib import Path
import torch.distributed as dist


class ModelSaver:
    """Handler that saves model checkpoints on a disk.

    The filename of the checkpoint is ``{filename_prefix}[_{score_name}]_{score:.4f}[_{epoch}].pt``

    (Optional) The filename of latest model is ``{filename_prefix}_latest.pt``
    (Optional) The filename of best model is ``{filename_prefix}_best.pt``

    We put the score first in the filename so that the files are sorted by score and tab completion works well.

    Args:
        dirname (str): Directory path where the checkpoint will be saved
        filename_prefix (str, optional): prefix for filename.
        score_name (str, optional): if not given, it will use "epoch" as default
        n_saved (int, optional): number of models to save.
        save_latest (bool, optional): if True, it will save the latest model. Defaults to ``False``
        save_best (bool, optional): if True, it will duplicate the best model with simple filename. Defaults to ``False``
        mode (str, optional): "max" or "min". If "max", the model with the highest score will be saved.
        atomic (bool, optional): if True, checkpoint is serialized to a temporary file, and then
            moved to final destination, so that files are guaranteed to not be damaged
            (for example if exception occurs during saving).
        create_dir (bool, optional): if True, will create directory ``dirname`` if it doesn't exist.
        require_empty (bool, optional): If True, will raise exception if there are any files in the
            directory ``dirname``.

    Examples:
        .. code-block:: python

            model_saver = ModelSaver(dirname="checkpoints", filename_prefix="model")
            for epoch in range(max_epochs):
                # train
                score = evaluate(model)
                model_saver.save(model, score, epoch)
            best_model = model_saver.last_checkpoint
            model.load_state_dict(torch.load(best_model))

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
        self.best_checkpoint = None
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
        self.last_checkpoint = self.history[-1][-1]


def load_json(fn):
    """Load json data from file

    Args:
        fn (str): file name

    Returns:
        Any: data
    """
    fn = os.path.expanduser(fn)
    with open(fn, "r") as f:
        data = json.load(f)
    return data


def save_json(fn, data, overwrite=False):
    """Save data in json format.

    Args:
        fn (str): file name
        data (Any): data to save
        overwrite (bool, optional): if True, will overwrite existing file. Defaults to False.
    """
    fn = os.path.expanduser(fn)
    if os.path.exists(fn) and not overwrite:
        print(f"{fn} already exists, skipping")
        return
    with open(fn, "w") as f:
        json.dump(data, f)


def load_csv(fn, num_skip=0, converter=None):
    """Load data in csv format.

    Args:
        fn (str): file name
        num_skip (int, optional): number of lines to skip. Defaults to 0.
        converter (Callable, optional): function to convert each element. Defaults to None.

    Returns:
        List: data
    """
    fn = os.path.expanduser(fn)
    with open(fn, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for _ in range(num_skip):
            next(reader)
        data = list(reader)
        if converter is not None:
            data = [list(map(converter, line)) for line in data]
    return data


def save_csv(fn, data, overwrite=False):
    """Save data in csv format.

    Args:
        fn (str): file name
        data (Any): data to save
        overwrite (bool, optional): if True, will overwrite existing file. Defaults to False.
    """
    fn = os.path.expanduser(fn)
    if os.path.exists(fn) and not overwrite:
        print(f"{fn} already exists, skipping")
        return
    with open(fn, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)
