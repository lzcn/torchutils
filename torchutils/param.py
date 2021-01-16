import operator
import os
import pprint
from numbers import Number
from typing import Any, Union

import attr
import torch.optim
import yaml

from . import misc

_factory = {}

_FACTORY_ATTR_NAME = "factory"


def toParam(kwargs):
    if isinstance(kwargs, dict) and "factory" in kwargs:
        return _factory[kwargs["factory"]](**kwargs)
    return kwargs


def filter(attribute: attr.Attribute, value: Any) -> bool:
    if attribute.name == _FACTORY_ATTR_NAME and value is None:
        return False
    return attribute.init


@attr.s
class Param(object):
    """Basic Param interface.

    Param class interface for all customizable classes.

    Only attributes whose repr is True will be displayed. The attributes defined
    with ``init=True`` is configurable, and ``repr=True`` printable.

    Note:
        Param is an abstract class whose implementation depends on attrs_.


    .. _attrs: https://www.attrs.org

    TODO:
        - `load` and `save` for interactiving with file.
        - support for different packages, e.g, yaml, json etc.
    """

    factory = attr.ib(default=None, kw_only=True, repr=False)

    def __init_subclass__(cls):
        super().__init_subclass__()
        _factory[cls.__name__] = cls

    def __str__(self):
        d = attr.asdict(self, filter=lambda attribute, _: attribute.repr is True)
        return self.__class__.__name__ + ":\n" + pprint.pformat(d)

    def asdict(self):
        """Return configurable attributes (e.g. whose init=True)."""
        return attr.asdict(self, filter=filter)

    def serialize(self):
        """Serialize configurable setttings to yaml foramt."""
        return yaml.dump(self.asdict())

    @classmethod
    def new(cls, value=None):
        """Return a new instance.

        If value is None, return an instance with default settings.

        TODO:
            A better way for sub-class initialization.

        """
        if value is None:
            return cls()
        elif isinstance(value, cls):
            return value
        return cls(**value)

    @classmethod
    def from_yaml(cls, f):
        """Return a new intance from yaml file

        Args:
            f (str): filename

        Returns:
            Param: new intance
        """
        with open(f, "r") as f:
            kwargs = yaml.load(f, Loader=misc.YAMLoader)
            param = cls(**kwargs)
        return param

    @classmethod
    def from_dict(cls, value=None):
        """Return a new intance from dictionary.

        If value is `None`, return `None`. So:

        - If a default instance is preferred, use `attr.ib(factory=dict, converter=cls.from_dict)`.
        - If no default instance is needed, use `attr.ib(default=None, converter=cls.from_dict)`.
        """
        if value is None or isinstance(value, cls):
            return value
        return cls(**value)


@attr.s
class DataReaderParam(Param):
    """Parameter class for :class:`torchutils.data.DataReader`.

    It supports an alternative way to initialize a DataReader intance::

        param = DataReaderParam(reader="ImageLMDB", path="data", data_transform="identity")
        reader = getReader(param=param)

    """

    reader = attr.ib()
    path = attr.ib()
    data_transform = attr.ib(default=None)

    @reader.validator
    def check(self, attribute, value):
        support = [
            "ImageLMDB",
            "ImagePIL",
            "TensorLMDB",
            "TensorPKL",
        ]
        if value not in support:
            raise ValueError("reader must be on of {}".format("|".joint(support)))


@attr.s
class SchedulerParam(Param):
    """Parameter class for :class:`torch.optim.lr_scheduler`

    Args:
        name (str): class name for lr_scheduler
        param (dict): class initialization param

    Examples:

        .. code-block::

            cfg = dict(
                name="StepLR",
                param=dict(
                    step_size=30,
                    gamma=0.1
                )
            )
            param = SchedulerParam(**cfg)

    """

    name = attr.ib(default="StepLR")
    param = attr.ib(factory=dict)

    def __attrs_post_init__(self):
        if self.name == "StepLR":
            default = dict(step_size=30, gamma=0.1)
        elif self.name == "ReduceLROnPlateau":
            default = dict(mode="max", cooldown=10, factor=0.1, patience=10, threshold=0.0001, verbose=True)
        elif self.name == "ExponentialLR":
            default = dict(gamma=0.5)
        else:
            raise KeyError
        # update default setting
        default.update(self.param)
        self.param = default


@attr.s
class OptimParam(Param):
    """Define parameters for optimizer

    Args:
        name : str
            which optimizer class to use, default SGD
        lr : [Number, list]
            number or a dict of {module name: lr} for each part in network
        weight_deacy : float
            same as learning rates
        param : dict
            specific setttings for each optimizer
        lr_scheduler : dict
            schedular for learning rate decay
    """

    name = attr.ib(default="SGD")
    lr = attr.ib(default=1e-3, repr=False)
    weight_decay = attr.ib(default=0, repr=False)
    param = attr.ib(factory=dict, repr=False)
    lr_scheduler: SchedulerParam = attr.ib(factory=dict, converter=SchedulerParam.from_dict)
    # hidden parameters
    groups = attr.ib(factory=dict, init=False)
    default = attr.ib(factory=dict, init=False)

    def __attrs_post_init__(self):
        # default settting for gradient  descent
        self.__grad_param__init__()
        # specific setting for different sub-modules
        self.__param_groups__init__()

    def __grad_param__init__(self):
        if self.name == "SGD":
            default = dict(momentum=0.9)
        elif self.name == "RMSprop":
            default = dict(alpha=0.99, eps=1e-08, momentum=0.9)
        elif self.name.startswith("Adam"):
            default = dict(betas=(0.9, 0.999), eps=1e-8)
        else:
            raise KeyError
        default.update(self.param)
        self.default = default

    def __param_groups__init__(self):
        """Parse setting for multiple group of parameters.

        If specific grouped learning rates or weight decay, then reaturn a group for
        each sub-module, else return None.
        """
        # learning rates and weight decay
        lrs, wds = self.lr, self.weight_decay
        self.use_group = True
        if isinstance(lrs, dict) and isinstance(wds, dict):
            assert wds.keys() == lrs.keys()
            self.groups = {k: dict(lr=v, weight_decay=wds[k]) for k, v in lrs.items()}
        elif isinstance(lrs, dict):
            self.groups = {k: dict(lr=v, weight_decay=wds) for k, v in lrs.items()}
        elif isinstance(wds, dict):
            self.groups = {k: dict(lr=lrs, weight_decay=wd) for k, wd in wds.items()}
        else:
            assert isinstance(lrs, Number), f"Learning rate must be a number or dict not a {type(lrs)}"
            assert isinstance(wds, Number), f"Weight decay must be a number or dict not a {type(wds)}"
            self.default.update(lr=lrs, weight_decay=wds)
            self.use_group = False

    def init_optimizer(self, net):
        """Init Optimizer for net."""
        # Optimizer and LR policy class
        grad_class = misc.get_named_class(torch.optim)[self.name]
        lr_class = misc.get_named_class(torch.optim.lr_scheduler)[self.lr_scheduler.name]
        # sub-module specific configuration
        if self.use_group:
            assert len(self.groups) > 0
            param_groups = []
            for name, param in self.groups.items():
                sub_module = operator.attrgetter(name)(net)
                param_group = dict(params=sub_module.parameters(), **param)
                param_groups.append(param_group)
            assert "lr" not in self.default
            optim = grad_class(param_groups, **self.default)
        else:
            optim = grad_class(net.parameters(), **self.default)
        # get instances
        lr_scheduler = lr_class(optim, **self.lr_scheduler.param)
        return optim, lr_scheduler


@attr.s
class IConfig(Param):
    """Configration interface.

    Attributes:
        epoch (int):  Number of epochs
        data_param (Param):
        train_data_param (dict):
        valid_data_param (dict):
        test_data_param (dict):
        net_param (Param):
        optim_param (OptimParam):
        summary_interval (int):
        display_interval (int):
        load_traiend (str):
        log_dir (str):
        log_level (str):
        log_file (str):
        gpus (int): which gpu to use

    Example:

        .. code-block:: python

            class Config(IConfig):
                data_param = attr.ib(factory=dict, covnert=DataParam.from_dict)
                net_param = attr.ib(factory=dict, covnert=NetParam.from_dict)

            config = Config.from_yaml(filename)
    """

    epochs: int = attr.ib(default=100)
    data_param: Param = attr.ib(factory=dict, converter=Param.from_dict)
    valid_data_param: Param = attr.ib(factory=dict)
    test_data_param: Param = attr.ib(factory=dict)
    train_data_param: Param = attr.ib(factory=dict)
    net_param: Param = attr.ib(factory=dict, converter=Param.from_dict)
    optim_param: OptimParam = attr.ib(default=None, converter=OptimParam.from_dict)
    summary_interval: int = attr.ib(default=10)
    display_interval: int = attr.ib(default=50)
    load_trained: str = attr.ib(default=None)
    log_dir: str = attr.ib(default=None)
    log_level: str = attr.ib(default="INFO")
    gpus: Union[int, list] = attr.ib(default=0)

    def __attrs_post_init__(self):
        self.train_data_param = attr.evolve(self.data_param, **self.train_data_param)
        self.valid_data_param = attr.evolve(self.data_param, **self.valid_data_param)
        self.test_data_param = attr.evolve(self.data_param, **self.test_data_param)
        # for gpus
        gpus = [self.gpus] if isinstance(self.gpus, int) else self.gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        self.gpus = list(range(len(gpus)))


@attr.s
class RunConfig(Param):
    """Configration interface for training/testing.

    Attributes:
        epoch (int):  Number of epochs
        data_param (Param):
        train_data_param (dict):
        valid_data_param (dict):
        test_data_param (dict):
        net_param (Param):
        optim_param (OptimParam):
        summary_interval (int):
        display_interval (int):
        load_traiend (str):
        log_dir (str):
        log_level (str):
        log_file (str):
        gpus (int): which gpu to use

    Example:

        .. code-block:: python

            class Config(IConfig):
                data_param = attr.ib(factory=dict, covnert=DataParam.from_dict)
                net_param = attr.ib(factory=dict, covnert=NetParam.from_dict)

            config = Config.from_yaml(filename)
    """

    epochs: int = attr.ib(default=100)
    data_param: Param = attr.ib(factory=dict, converter=toParam)
    valid_data_param: Param = attr.ib(factory=dict)
    test_data_param: Param = attr.ib(factory=dict)
    train_data_param: Param = attr.ib(factory=dict)
    net_param: Param = attr.ib(factory=dict, converter=toParam)
    optim_param: OptimParam = attr.ib(default=None, converter=OptimParam.from_dict)
    summary_interval: int = attr.ib(default=10)
    display_interval: int = attr.ib(default=50)
    load_trained: str = attr.ib(default=None)
    log_dir: str = attr.ib(default=None)
    log_level: str = attr.ib(default="INFO")
    gpus: Union[int, list] = attr.ib(default=0)

    def __attrs_post_init__(self):
        if isinstance(self.data_param, Param):
            self.train_data_param = attr.evolve(self.data_param, **self.train_data_param)
            self.valid_data_param = attr.evolve(self.data_param, **self.valid_data_param)
            self.test_data_param = attr.evolve(self.data_param, **self.test_data_param)
        # for gpus
        gpus = [self.gpus] if isinstance(self.gpus, int) else self.gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        self.gpus = list(range(len(gpus)))
