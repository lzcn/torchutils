import logging
from collections import deque

import numpy as np


class _WindowMeter(object):
    def __init__(self, win_size=50):
        self.val = float("nan")
        self.win_size = win_size
        self._weights = deque(maxlen=win_size)
        self._values = deque(maxlen=win_size)

    def reset(self):
        self.val = float("nan")
        self._values.clear()
        self._weights.clear()

    @property
    def avg(self):
        try:
            return sum(self._values) / sum(self._weights)
        except ZeroDivisionError:
            return float("nan")

    def update(self, value, weight=1.0):
        self.val = value
        self._values.append(value * weight)
        self._weights.append(weight)

    def __repr__(self):
        return "{:.4f} ({:.4f})".format(self.val, self.avg)


class _AverageMeter(object):
    def __init__(self):
        self.val = float("nan")
        self._cum_val = 0.0
        self._cum_weight = 0.0

    @property
    def avg(self):
        try:
            return self._cum_val / self._cum_weight
        except ZeroDivisionError:
            return float("nan")

    def update(self, value, weight=1.0):
        self.val = val
        self._cum_val += val * weight
        self._cum_weight += weight

    def __repr__(self):
        return "{:.4f} ({:.4f})".format(self.val, self.avg)


def _factory(win_size):
    return _AverageMeter() if win_size == 0 else _WindowMeter(win_size)


class Meter(object):
    """Meter for history values.

    Args:
        win_size ([int]): window size for moving average. If `win_size==0`, then all
            values will be considered.



    """

    def __init__(self, win_size=10):
        self.win_size = win_size
        self._meter = _factory(win_size)

    @property
    def val(self):
        """Get the current value."""
        return self._meter.val

    @property
    def avg(self):
        """Get the averaged value."""
        return self._meter.avg

    def update(self, value, weight=1.0):
        """Update the meter."""
        self._meter.update(value, weight)

    def __repr__(self):
        return "{:.4f} ({:.4f})".format(self._meter.val, self._meter.avg)


class BundleMeter(object):
    """Manage a bunle of meters with the same window size.

    Args:
        win_size (int): window size for all meters.

    Example:

    .. code-block:: python

        # history tracer with window size 10
        tracer = BundleMeter(10)
        # update data
        tracer.update(data={'loss':0.1, 'accuracy':0.8}, score=0.3)
        # show current status
        tracer.logging()

    """

    def __init__(self, win_size):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.win_size = win_size
        self._meters = dict()

    @property
    def val(self) -> dict:
        """Return the current value."""
        return {k: v.val for k, v in self._meters.items()}

    @property
    def avg(self) -> dict:
        """Return the averaged value."""
        return {k: v.avg for k, v in self._meters.items()}

    def _get_meter(self, key):
        if key not in self._meters:
            self._meters[key] = _factory(self.win_size)
        return self._meters[key]

    def update(self, weight=1.0, data=None, **kwargs):
        """Update meters.

        If the metric is new, it will produce a new meter for the metric.
        """
        data = dict() if data is None else data
        kwargs.update(**data)
        for key, value in kwargs.items():
            self._get_meter(key).update(value, weight=weight)

    def logging(self):
        for k, m in self._meters.items():
            self.logger.info("-------- %s: %s", k, m)

    def __repr__(self):
        result = ""
        for k, v in self._meters.items():
            result += "-------- {}:{}\n".format(k, v)
        return result


class GroupMeter(object):
    """Class for history tracer with different window size.

    Args:
        win_size (dict): a set the window size for meters for each group

    Example:

    .. code-block:: python

        tracer = GroupMeter(train=10, test=1)
        tracer.update("train", data={'loss':0.1, 'accuracy':0.8})
        tracer.update("test", data={'loss':0.2, 'accuracy':0.7})
        tracer.logging() # print(tracer)

    """

    def __init__(self, **win_size):
        # create meter factories for each group
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._win_size = win_size
        self._meters = {g: dict() for g in win_size.keys()}

    def _get_meter(self, group, key):
        if key not in self._meters[group]:
            self._meters[group][key] = _factory(self._win_size[group])
        return self._meters[group][key]

    def update(self, group, weight=1.0, data=None, **kwargs):
        data = dict() if data is None else data
        kwargs.update(**data)
        for key, value in kwargs.items():
            self._get_meter(group, key).update(value, weight=weight)

    def logging(self, group=None):
        if group:
            for k, m in self._meters[group].items():
                self.logger.info("-------- %s: %s", k, m)
        else:
            for group, history in self._meters.items():
                self.logger.info("-------- Group: %s", group)
                for k, m in history.items():
                    self.logger.info("-------- %s: %s", k, m)

    def __repr__(self):
        str_ = ""
        for group, history in self._meters.items():
            str_ += "-------- Group: {}\n".format(group)
            if len(history) == 0:
                str_ += "-------- (Empty)\n"
            for k, m in history.items():
                str_ += "-------- {}: {}\n".format(k, m)
        return str_
