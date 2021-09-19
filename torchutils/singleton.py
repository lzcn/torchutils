def singleton(cls):
    """Decorator for singleton class.

    Example:
        .. code-block:: python

            @torchutils.singleton
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
