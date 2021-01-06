__all__ = ["singleton"]


def singleton(cls):
    """Decorator for singleton class.

    Example:

    .. code-block:: python

        @utils.singleton
        class A(object):
           ...
        x = A()
        y = A()
        assert id(x) == id(y)

    """
    _instance = {}

    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner


if __name__ == "__main__":

    @singleton
    class cls(object):
        def __init__(self):
            print("Initilize a new object.")

    x = cls()
    y = cls()
    assert id(x) == id(y)
