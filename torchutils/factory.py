from typing import Union

from torch import nn
from torch.utils import data

from torchutils.param import Param

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


def get_dataloader(name_or_param: Union[str, Param], **kwargs) -> data.DataLoader:
    """Return registered dataloader."""
    if isinstance(name_or_param, Param):
        return _dataloader_regisrty[name_or_param.name](name_or_param)
    return _dataloader_regisrty[name_or_param](**kwargs)


def get_dataset(name_or_param: Union[str, Param], **kwargs) -> data.Dataset:
    """Return registered dataset."""
    if isinstance(name_or_param, Param):
        return _dataset_registry[name_or_param.name](name_or_param)
    return _dataset_registry[name_or_param](**kwargs)


def get_module(name_or_param: Union[str, Param], **kwargs) -> nn.Module:
    """Return registered module."""
    if isinstance(name_or_param, Param):
        return _module_registry[name_or_param.name](name_or_param)
    return _module_registry[name_or_param](**kwargs)


def get_named_dataloaders():
    """Return all registered dataloaders with name."""
    return _dataloader_regisrty.copy()


def get_named_datasets():
    """Return all registered datasets with name."""
    return _dataset_registry.copy()


def get_named_modules():
    """Return all registered modules with name."""
    return _module_registry.copy()
