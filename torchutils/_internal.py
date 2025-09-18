def set_module(module: str):
    """Private decorator for overriding ``__module__`` on a function or class.

    This is used to make internal functions or classes appear as part of a public
    API in documentation and introspection tools.

    If used on a class, the original module can optionally be preserved in
    ``_module_source``.

    Parameters
    ----------
    module : str
        The name of the module to set (e.g., ``'torchutils'`` or ``'numpy'``).

    Returns
    -------
    Callable
        A decorator that sets ``__module__`` on the decorated object.

    Examples
    --------
    >>> @set_module('numpy')
    ... def example():
    ...     pass
    >>> example.__module__
    'numpy'

    Notes
    -----
    This function comes from :mod:`numpy._utils`.
    """

    def decorator(func):
        if module is not None:
            if isinstance(func, type):
                try:
                    func._module_source = func.__module__
                except AttributeError:
                    pass
            try:
                func.__module__ = module
            except (TypeError, AttributeError):
                pass
        return func

    return decorator
