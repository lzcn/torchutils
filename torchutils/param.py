import yaml
import attr
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
    """

    def __str__(self):
        d = attr.asdict(self, filter=lambda attribute, _: attribute.repr is True)
        return self.__class__.__name__ + ":\n" + misc.format_display(d)

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
