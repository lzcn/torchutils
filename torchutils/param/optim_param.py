from logging import warning
import operator
from numbers import Number

import attr
import torch.optim

from .. import misc
from .param import Param


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
            num_sumodule = len(list(net.children()))
            if num_sumodule != len(self.groups):
                warning(f"Number of sub-modules {num_sumodule} not equal to number of groups {len(self.groups)}")
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
