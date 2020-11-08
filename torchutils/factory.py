from torch import nn
from torch.utils import data

__all__ = [
    "DataLoader",
    "Dataset",
    "Module",
    "get_dataloader",
    "get_dataset",
    "get_module",
    "get_named_dataloaders",
    "get_named_datasets",
    "get_named_modules",
]

_dataloader_regisrty = {}
_dataset_registry = {}
_module_registry = {}


class DataLoader(data.DataLoader):
    r"""Wrapped base class for :class:`torch.utils.data.DataLoader`.

    Subclass will be registered by it's name::
    """

    def __init_subclass__(cls):
        super().__init_subclass__()
        _dataloader_regisrty[cls.__name__] = cls


class Dataset(data.Dataset):
    r"""Wrapped base class for :class:`torch.utils.data.Dataset`.

    Subclass will be registered by it's name::
    """

    def __init_subclass__(cls):
        super().__init_subclass__()
        _dataset_registry[cls.__name__] = cls


class Module(nn.Module):
    r"""Wrapped base class for :class:`torch.nn.Module`.

    Subclass will be registered by it's name::


        # simple_model.py
        import torchutils.factory as factory
        import torch.nn.functional as F

        class SimpleModel(factory.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, 1)
                self.conv2 = nn.Conv2d(64, 64, 3, 1)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))


        # main.py
        import torchutils.factory as factory
        net = factory.get_module["SimpleModel"]()

    """

    def __init_subclass__(cls):
        super().__init_subclass__()
        _module_registry[cls.__name__] = cls


def get_dataloader(name):
    """Return registered dataloader."""
    return _dataloader_regisrty[name]


def get_dataset(name):
    """Return registered dataset."""
    return _dataset_registry[name]


def get_module(name):
    """Return registered module."""
    return _module_registry[name]


def get_named_dataloaders():
    """Return all registered dataloaders with name."""
    return _dataloader_regisrty.copy()


def get_named_datasets():
    """Return all registered datasets with name."""
    return _dataset_registry.copy()


def get_named_modules():
    """Return all registered modules with name."""
    return _module_registry.copy()
