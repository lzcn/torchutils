import logging
import pprint
from typing import Any

import attr
import yaml

from .. import misc

_factory = {}

_FACTORY_ATTR_NAME = "factory"
LOGGER = logging.getLogger(__name__)


def _configurable_filter(attribute: attr.Attribute, value: Any) -> bool:
    if attribute.name == _FACTORY_ATTR_NAME and value is None:
        return False
    return attribute.init


def _printable_filter(attribute: attr.Attribute, value: Any) -> bool:
    if attribute.name == _FACTORY_ATTR_NAME and value is None:
        return False
    return attribute.repr


def _format_display(opt: dict, num=1):
    indent = "  " * num
    string = ""
    for k, v in opt.items():
        if v is None:
            continue
        if isinstance(v, dict):
            string += "{}{} : {{\n".format(indent, k)
            string += _format_display(v, num + 2)
            string += "{}}},\n".format(indent)
        elif isinstance(v, list):
            string += "{}{} : ".format(indent, k)
            one_line = ",".join(map(str, v))
            if len(one_line) < 87:
                string += "[" + one_line + "]\n"
            else:
                prefix = "  " + indent
                string += "[\n"
                for i in v:
                    string += "{}{},\n".format(prefix, i)
                string += "{}]\n".format(indent)
        else:
            string += "{}{} : {},\n".format(indent, k, v)
    return string


@attr.s
class Param(object):
    r"""Basic parameter class.

    This class supports hierarchy structure and its subclass will be registered by class
    name. The implementation of :meth:`Param` depends on attrs_. For hierarchical
    attribute, i.e., a subclass of :class:`Param`, you can use :meth:`toParam`
    as the converter::

        attr.ib(converter=toParam)

    Note:

        Only attributes whose repr is True will be displayed. The attributes defined
        with ``init=True`` is configurable, and ``repr=True`` printable.


    Attributes:
        factory (str): class name for used for initialization.

    .. _attrs: https://www.attrs.org

    TODO:

        serialization：
            - `load` and `save` for interactiving with file.
            - support for different packages, e.g, yaml, json etc.
    """

    # class name for param
    factory: str = attr.ib(default=None, kw_only=True)

    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls.__name__ in _factory:
            LOGGER.warning("%s is already used", cls.__name__)
        _factory[cls.__name__] = cls

    def __str__(self):
        d = attr.asdict(self, filter=_printable_filter)
        return self.__class__.__name__ + ":\n" + _format_display(d)

    def asdict(self):
        r"""Return configurable attributes (e.g. whose init=True)."""
        return attr.asdict(self, filter=_configurable_filter)

    def serialize(self):
        r"""Serialize configurable settings to yaml format."""
        return yaml.dump(self.asdict())

    @classmethod
    def evolve(cls, inst=None, **changes):
        r"""Create a new instance based on given instance and changes.

        directly calling evolve() will create a new instance with default values.

        Examples::

                param = Param.evolve() # default instance
                param = Param.evolve(x=1) # default instance with partial keywords.
                param = Param.evolve(param, x=2) # instance with changes applied on param.

        Args:
            inst (Param, optional): instance of given class. Defaults to ``None``.
            changes: keywords changes

        Returns:
            Param: new instance
        """
        inst = cls() if inst is None else inst
        return attr.evolve(inst, **changes)

    @classmethod
    def new(cls, value=None):
        r"""Return a new instance.

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
    def from_yaml(cls, fn):
        r"""Return a new instance from a yaml file.

        Args:
            fn (str): filename

        Returns:
            Param: new instance
        """
        with open(fn, "r") as fn:
            kwargs = yaml.load(fn, Loader=misc.YAMLoader)
            param = cls(**kwargs)
        return param

    @classmethod
    def from_dict(cls, value=None):
        r"""Return a new instance from a dictionary.

        Args:
            value (Any, optional): If value is ``None``, return ``None``. Defaults to None.

        Used as the convert for :meth:`attr.ib`:
            - If a default instance is preferred, use ``attr.ib(factory=dict, converter=cls.from_dict)``.
            - If no default instance is needed, use ``attr.ib(default=None, converter=cls.from_dict)``.

        """
        if value is None or isinstance(value, cls):
            return value
        return cls(**value)


def to_param(value) -> Param:
    r"""Converter for param

    If the input is an instance of dict and value["factory"] is a subclass of
    :class:`Param`, then return an instance, otherwise, input itself will be returned.

    Args:
        value (Any): configurations

    Examples::

        value = {
            "factory": "OptimParam",
            # ...,
        }
        param = to_param(value)
        assert isinstance(param, OptimParam)

    """
    class_name = value.get(_FACTORY_ATTR_NAME)
    if isinstance(value, dict) and class_name in _factory:
        return _factory[class_name](**value)
    return value
