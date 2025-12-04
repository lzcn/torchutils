def format_display(obj, indent_level=1, indent_char=" "):
    """Convert nested structures to a nicely formatted string.

    Args:
        obj: Object to format (dict, list, or other).
        indent_level: Current indentation level.
        indent_char: Character(s) to use for indentation.

    Returns:
        Formatted string representation.

    Example:
        >>> data = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        >>> print(format_display(data))
        {
         a: 1,
         b: [2, 3],
         c: {
          d: 4,
         },
        }
    """
    indent = indent_char * indent_level

    if isinstance(obj, dict):
        items = [
            f"{k}: {format_display(v, indent_level + 1, indent_char)}"
            for k, v in obj.items()
        ]
        if sum(map(len, items)) < 10:
            return "{" + ", ".join(items) + "}"
        result = "{\n"
        for item in items:
            result += f"{indent}{item},\n"
        result += indent_char * (indent_level - 1) + "}"
        return result

    elif isinstance(obj, list):
        items = [format_display(v, indent_level + 1, indent_char) for v in obj]
        if sum(map(len, items)) < 10:
            return "[" + ", ".join(items) + "]"
        result = "[\n"
        for item in items:
            result += f"{indent}{item},\n"
        result += indent_char * (indent_level - 1) + "]"
        return result

    else:
        return str(obj)


def get_public_classes(module):
    """Return public classes defined in a module as a dict.

    Args:
        module: Python module object.

    Returns:
        Dict mapping class names to class objects.

    Example:
        >>> import torch.nn as nn
        >>> classes = get_public_classes(nn)
        >>> 'Linear' in classes
        True
    """
    from inspect import isclass

    return {
        k: v for k, v in module.__dict__.items() if isclass(v) and not k.startswith("_")
    }


def get_public_functions(module):
    """Return public functions defined in a module as a dict.

    Args:
        module: Python module object.

    Returns:
        Dict mapping function names to function objects.

    Example:
        >>> import math
        >>> funcs = get_public_functions(math)
        >>> 'sqrt' in funcs
        True
    """
    from inspect import isfunction

    return {
        k: v
        for k, v in module.__dict__.items()
        if isfunction(v) and not k.startswith("_")
    }
