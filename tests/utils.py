"""Test helpers for device-aware scenarios and shared model fixtures."""

import unittest

import torch
from torch import nn

__all__ = ["get_available_device", "requires_cuda", "SimpleModel"]


def get_available_device() -> torch.device:
    """Return the best available torch device, preferring CUDA when present."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available.")


class SimpleModel(nn.Module):
    """Minimal feed-forward network used across test cases."""

    def __init__(self, in_features: int = 10, out_features: int = 1) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fc(x)
