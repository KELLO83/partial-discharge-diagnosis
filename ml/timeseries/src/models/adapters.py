"""Input adapters for official time-series model wrappers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_time_axis_channel_first(x: torch.Tensor, target_length: int) -> torch.Tensor:
    """Resize (batch, channel, time) to a target time length."""
    if x.shape[-1] == target_length:
        return x
    return F.interpolate(x, size=target_length, mode="linear", align_corners=False)


def resize_time_axis_time_first(x: torch.Tensor, target_length: int) -> torch.Tensor:
    """Resize (batch, time, channel) to a target time length."""
    if x.shape[1] == target_length:
        return x
    return resize_time_axis_channel_first(x.transpose(1, 2), target_length).transpose(1, 2)
