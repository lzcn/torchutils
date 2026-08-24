from typing import Any

import torch


def to(data: Any, device: str | torch.device = "cuda") -> Any:
    """Recursively move tensors in a nested structure to the given device.

    Dicts, lists and tuples are traversed; non-tensor leaves pass through.

    Args:
        data: Tensor, or nested structure (dict / list / tuple) of tensors.
        device: Target device, e.g. "cuda" or "cpu".

    Returns:
        Data placed on the specified device.
    """
    match data:
        case dict():
            return {k: to(v, device) for k, v in data.items()}
        case list():
            return [to(v, device) for v in data]
        case tuple():
            return tuple(to(v, device) for v in data)
        case torch.Tensor():
            return data.to(device, non_blocking=True) if device != "cpu" else data.cpu()
        case _:
            return data
