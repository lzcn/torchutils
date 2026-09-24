from typing import Any

import torch


def to(data: Any, device: str | torch.device = "cuda", non_blocking: bool = True) -> Any:
    """Recursively move tensors in a nested structure to the given device.

    Dicts, lists and tuples are traversed; non-tensor leaves pass through.
    Namedtuples keep their type.

    Args:
        data: Tensor, or nested structure (dict / list / tuple) of tensors.
        device: Target device, e.g. "cuda" or "cpu".
        non_blocking: Try async transfer (effective with pinned memory).

    Returns:
        Data placed on the specified device.
    """
    match data:
        case dict():
            return {k: to(v, device, non_blocking) for k, v in data.items()}
        case list():
            return [to(v, device, non_blocking) for v in data]
        case tuple():
            items = (to(v, device, non_blocking) for v in data)
            return type(data)(*items) if hasattr(data, "_fields") else tuple(items)
        case torch.Tensor():
            return data.to(device, non_blocking=non_blocking)
        case _:
            return data
