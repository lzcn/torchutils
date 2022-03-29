import csv
import json
import os
from typing import Optional

from ignite import handlers
from torch import nn
from torch.types import Number


class ModelSaver(object):
    """Handler that saves model checkpoints on a disk.


    This class uses :class:`~ignite.handlers.DiskSaver` as saver.

    Examples:

        .. code-block:: python

            from torchutils.io import ModelSaver
            model_saver = ModelSaver(dirname="checkpoints", filename_prefix="model")
            for epoch in range(max_epochs):
                // train
                score = evaluate(model)
                model_saver.save(model, score, epoch)
            best_model = model_saver.last_checkpoint
            model.load_state_dict(torch.load(best_model))

    Args:
        dirname (str): Directory path where the checkpoint will be saved
        filename_prefix (str, optional): prefix for filename.
        score_name (str, optional): if not given, it will use "epoch" as default
        n_saved (int, optional): number of models to save.
        save_laset (bool, optional): if True, it will save the latest model. Defaults to False
        mode (str, optional): "max" or "min". If "max", the model with the highest score will be saved.
        atomic (bool, optional): if True, checkpoint is serialized to a temporary file, and then
            moved to final destination, so that files are guaranteed to not be damaged
            (for example if exception occurs during saving).
        create_dir (bool, optional): if True, will create directory ``dirname`` if it doesn't exist.
        require_empty (bool, optional): If True, will raise exception if there are any files in the
            directory ``dirname``.


    """

    def __init__(
        self,
        dirname: str,
        filename_prefix: str = None,
        score_name: str = None,
        n_saved: Optional[int] = 5,
        save_laset: bool = False,
        mode="max",
        atomic: bool = True,
        create_dir: bool = True,
        require_empty: bool = True,
    ):
        super().__init__()
        self.dirname = dirname
        self.filename_predfix = filename_prefix
        self.score_name = score_name
        self.n_saved = n_saved
        self.save_latest = save_laset
        self.mode = mode
        self.history = []
        self.best_checkpoint = None
        self.saver = handlers.DiskSaver(dirname, atomic=atomic, create_dir=create_dir, require_empty=require_empty)

    def filename(self, score: float = None, epoch: int = None) -> str:
        prefix = f"{self.filename_predfix}_" if self.filename_predfix else ""
        score_name = f"{self.score_name}_" if self.score_name else ""
        if score is None and epoch is None:
            filename = f"{prefix}latest.pt"
        elif score is None and epoch is not None:
            filename = f"{prefix}epoch_{epoch}.pt"
        elif score is not None and epoch is None:
            filename = f"{prefix}{score_name}{score:.4f}.pt"
        else:
            filename = f"{prefix}{score_name}{score:.4f}_epoch_{epoch}.pt"
        return filename

    def save(self, model: nn.Module, score, epoch=None):
        """Save model checkpoint.

        The format of the filename is ``{filename_prefix}[_{score_name}]_{score:.4f}[_{epoch}].pt``
        where [_{score_name}] and [_{epoch}] are optional.

        Args:
            model (nn.Module): model to save
            score (float, Optional): score
            epoch (Number, Optional): current epoch
        """
        state_dict = model.state_dict()
        if self.save_latest:
            self.saver(state_dict, filename=self.filename())
        filename = self.filename(score, epoch)
        if len(self.history) < self.n_saved:  # pool not full
            self.saver(state_dict, filename=filename)
            self.history.append((score, filename))
        elif score > self.history[0][0]:  # replace the worst model
            self.saver(state_dict, filename=filename)
            try:
                self.saver.remove(self.history[0][1])
            except FileNotFoundError:
                pass
            self.history[0] = (score, filename)
        else:  # score is not better than the worst model
            pass
        # sort the history
        if self.mode == "max":
            self.history.sort(key=lambda x: x[0])
        else:
            self.history.sort(key=lambda x: x[0], reverse=True)
        self.last_checkpoint = os.path.join(self.dirname, self.history[-1][-1])


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
