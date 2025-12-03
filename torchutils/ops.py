from typing import Any, Mapping, Union

import torch


def to(data: Any, device: Union[str, torch.device] = "cuda") -> Any:
    """Recursively move data to the specified device.

    Args:
        data: Tensor, or nested structure of tensors (dict, list, tuple).
        device: Target device (e.g., "cuda", "cpu").

    Returns:
        Data placed on the specified device.
    """
    if isinstance(data, Mapping):
        return {k: to(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True) if device != "cpu" else data.cpu()
    elif isinstance(data, str):
        return data
    else:
        raise TypeError(f"Unsupported data type for device transfer: {type(data).__name__}")

