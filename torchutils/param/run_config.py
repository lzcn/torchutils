import logging
import os
from typing import Union

import attr

from .. import factory, misc
from .optim_param import OptimParam
from .param import Param, toParam

LOGGER = logging.getLogger(__name__)


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

    def get_net(self, training=False, cuda=False, **changes):
        net_param = attr.evolve(self.net_param, **changes)
        net = factory.get_module(net_param)
        LOGGER.info("Get network:\n{}".format(net_param))
        if self.load_trained:
            net = misc.load_pretrained(net, self.load_trained)
        if training:
            net.train()
        else:
            net.eval()
        if cuda:
            net.cuda()
        return net

    def get_train_loader(self, **changes):
        data_param = attr.evolve(self.train_data_param, **changes)
        LOGGER.info("Get training dataloader:\n{}".format(data_param))
        dataloader = factory.get_dataloader(data_param)
        return dataloader

    def get_valid_loader(self, **changes):
        data_param = attr.evolve(self.valid_data_param, **changes)
        LOGGER.info("Get validation dataloader:\n{}".format(data_param))
        dataloader = factory.get_dataloader(data_param)
        return dataloader

    def get_test_loader(self, **changes):
        data_param = attr.evolve(self.test_data_param, **changes)
        LOGGER.info("Get testing dataloader:\n{}".format(data_param))
        dataloader = factory.get_dataloader(data_param)
        return dataloader
