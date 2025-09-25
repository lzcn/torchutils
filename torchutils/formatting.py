"""Formatting helpers exposed by torchutils."""

from ._internal import set_module


@set_module("torchutils")
def format_display(opt, num=1, symbol=" "):
    """Convert nested structures to a nicely formatted string."""

    indent = symbol * num
    if isinstance(opt, dict):
        repr_list = [f"{k}: {format_display(v, num + 1, symbol)}" for k, v in opt.items()]
        lsign = "{"
        rsign = "}"
        if sum(map(len, repr_list)) < 10:
            string = lsign + ", ".join(repr_list) + rsign
        else:
            string = lsign + "\n"
            for repr in repr_list:
                string += f"{indent}{repr},\n"
            string += symbol * (num - 1) + rsign
    elif isinstance(opt, list):
        repr_list = [format_display(v, num + 1, symbol) for v in opt]
        lsign = "["
        rsign = "]"
        if sum(map(len, repr_list)) < 10:
            string = lsign + ", ".join(repr_list) + rsign
        else:
            string = lsign + "\n"
            for repr in repr_list:
                string += f"{indent}{repr},\n"
            string += symbol * (num - 1) + rsign
    else:
        string = str(opt)
    return string
