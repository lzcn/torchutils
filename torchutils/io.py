import csv
import json
import os
from typing import Optional

from torch import nn
from torch.types import Number

from ignite import handlers


class ModelSaver(object):
    """Handler that saves model checkpoints on a disk.


    This class uses :class:`~ignite.handlers.DiskSaver` as saver.

    Args:
        dirname (str): Directory path where the checkpoint will be saved
        filename_prefix (str, optional): prefix for filename.
        score_name (str, optional): if not given, it will use "epoch" as default
        n_saved (int, optional): number of models to save.
        atomic (bool, optional): if True, checkpoint is serialized to a temporary file, and then
            moved to final destination, so that files are guaranteed to not be damaged
            (for example if exception occurs during saving).
        create_dir (bool, optional): if True, will create directory ``dirname`` if it doesnt exist.
        require_empty (bool, optional): If True, will raise exception if there are any files in the
            directory ``dirname``.


    """

    def __init__(
        self,
        dirname: str,
        filename_prefix: str = None,
        score_name: str = None,
        n_saved: Optional[int] = 5,
        atomic: bool = True,
        create_dir: bool = True,
        require_empty: bool = True,
    ):
        super().__init__()
        self.saver = handlers.DiskSaver(dirname, atomic=atomic, create_dir=create_dir, require_empty=require_empty)
        self.filename_predfix = filename_prefix
        self.score_name = score_name
        self.n_saved = n_saved
        self.history = []

    def filename(self, score, epoch=None):
        name = f"{self.score_name}_{score:.4f}_" if self.score_name else f"{score:.4f}_"
        suffix = f"epoch_{epoch}" if epoch is not None else ""
        prefix = f"{self.filename_predfix}_" if self.filename_predfix else ""
        return f"{prefix}{name}{suffix}.pt"

    def save(self, model: nn.Module, score: Number, epoch=None):
        """Save model given score or epoch.

        Args:
            model (nn.Module): model to save
            score (Number): score or epoch
        """
        state_dict = model.state_dict()
        filename = self.filename(score, epoch)
        if len(self.history) < self.n_saved:  # history not full
            self.saver(state_dict, filename=filename)
            self.history.append((score, filename))
        elif score > self.history[0][0]:  # top-n best results
            self.saver(state_dict, filename=filename)
            _, last_file = self.history[0]
            self.saver.remove(last_file)
            self.history[0] = (score, filename)
        else:  # do not save
            pass
        self.history = sorted(self.history, key=lambda x: x[0])

    @property
    def last_checkpoint(self):
        if len(self.history) < 1:
            return None
        return self.history[-1][-1]


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
