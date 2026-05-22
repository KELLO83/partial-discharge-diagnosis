"""Preprocessing helpers for partial discharge signals."""

from __future__ import annotations

import torch


def ensure_channel_first(batch: torch.Tensor) -> torch.Tensor:
    """Return batch as (batch, pseudo_channel, time)."""
    if batch.ndim != 3:
        raise ValueError(f"Expected 3D batch, got shape={tuple(batch.shape)}")
    if batch.shape[1] == 20:
        return batch
    if batch.shape[2] == 20:
        return batch.transpose(1, 2)
    raise ValueError(f"Cannot infer pseudo-channel axis from shape={tuple(batch.shape)}")


def ensure_time_first(batch: torch.Tensor) -> torch.Tensor:
    """Return batch as (batch, time, pseudo_channel)."""
    if batch.ndim != 3:
        raise ValueError(f"Expected 3D batch, got shape={tuple(batch.shape)}")
    if batch.shape[2] == 20:
        return batch
    if batch.shape[1] == 20:
        return batch.transpose(1, 2)
    raise ValueError(f"Cannot infer pseudo-channel axis from shape={tuple(batch.shape)}")
