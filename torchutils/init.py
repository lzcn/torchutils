from functools import partial

from torch import nn


def xavier_normal_weight(gain=1.0):
    """apply xavier normal weight initialization."""

    def _init_func(module, gain):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    return partial(_init_func, gain=gain)


def xavier_uniform_weight(gain=1.0):
    """apply xavier uniform weight initialization."""

    def _init_func(module, gain):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    return partial(_init_func, gain=gain)


def kaiming_normal_weight(a=0, mode="fan_in", nonlinearity="leaky_relu"):
    """apply kaiming normal weight initialization."""

    def _init_func(module, a, mode, nonlinearity):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    return partial(_init_func, a=a, mode=mode, nonlinearity=nonlinearity)


def kaiming_uniform_weight(a=0, mode="fan_in", nonlinearity="leaky_relu"):
    """apply kaiming uniform weight initialization."""

    def _init_func(module, a, mode, nonlinearity):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    return partial(_init_func, a=a, mode=mode, nonlinearity=nonlinearity)
