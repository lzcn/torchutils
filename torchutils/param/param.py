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


@attr.s
class Param(object):
    r"""Basic parameter class.

    Any subclass will be registered. For attribute that is a subclass of :class:`Param`
    use :meth:`toParam` as the converter, i.e. ``attr.ib(converter=toParam)``

    Only attributes whose repr is True will be displayed. The attributes defined
    with ``init=True`` is configurable, and ``repr=True`` printable.

    Attributes:
        factory (str): class name for used for initialization.

    Note:
        Param is an abstract class whose implementation depends on attrs_.


    .. _attrs: https://www.attrs.org

    TODO:

        serialization：
            - `load` and `save` for interactiving with file.
            - support for different packages, e.g, yaml, json etc.

    TODO:

        replace Param with FactoryParam

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
        return self.__class__.__name__ + ":\n" + pprint.pformat(d)

    def asdict(self):
        r"""Return configurable attributes (e.g. whose init=True)."""
        return attr.asdict(self, filter=_configurable_filter)

    def serialize(self):
        r"""Serialize configurable setttings to yaml foramt."""
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
        r"""Return a new intance from yaml file

        Args:
            fn (str): filename

        Returns:
            Param: new intance
        """
        with open(fn, "r") as fn:
            kwargs = yaml.load(fn, Loader=misc.YAMLoader)
            param = cls(**kwargs)
        return param

    @classmethod
    def from_dict(cls, value=None):
        r"""Return a new intance from dictionary.

        If value is ``None``, return ``None``. So:

        - If a default instance is preferred, use ``attr.ib(factory=dict, converter=cls.from_dict)``.
        - If no default instance is needed, use ``attr.ib(default=None, converter=cls.from_dict)``.

        """
        if value is None or isinstance(value, cls):
            return value
        return cls(**value)


def toParam(kwargs) -> Param:
    """factory converter.
    """
    if isinstance(kwargs, dict) and "factory" in kwargs:
        return _factory[kwargs["factory"]](**kwargs)
    return kwargs
