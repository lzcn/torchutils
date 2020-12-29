import operator
from numbers import Number
import pprint
import torch.optim
import attr
import yaml

from . import misc


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

    def __str__(self):
        d = attr.asdict(self, filter=lambda attribute, _: attribute.repr is True)
        return self.__class__.__name__ + ":\n" + pprint.pformat(d)

    def asdict(self):
        """Return configurable attributes (e.g. whose init=True)."""
        return attr.asdict(self, filter=lambda attribute, _: attribute.init is True)

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
            kwargs = yaml.load(f, Loader=yaml.FullLoader)
            param = cls(**kwargs)
        return param


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
    lr_scheduler: SchedulerParam = attr.ib(default=None, converter=SchedulerParam.new)
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
        elif self.name == "Adabound":
            default = dict(betas=(0.9, 0.999), final_lr=0.1, gamme=1e-3, eps=1e-8)
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
