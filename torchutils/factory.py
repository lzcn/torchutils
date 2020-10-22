from torch import nn

__all__ = ["Module", "get_named_modules", "get_module"]

_module_registry = {}


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


def get_named_modules():
    return _module_registry.copy()


def get_module(name):
    return _module_registry[name]
