import attr

from .param import Param


@attr.s
class DataReaderParam(Param):
    """Parameter class for :class:`torchutils.data.DataReader`.

    It supports an alternative way to initialize a DataReader instance::

        param = DataReaderParam(reader="ImageLMDB", path="data", data_transform="identity")
        reader = getReader(param=param)

    """

    reader = attr.ib(default="Dummy")
    path = attr.ib(default=None)
    data_transform = attr.ib(default=None)
