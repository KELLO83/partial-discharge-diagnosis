"""Common PyTorch model wrapper interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class BaseTimeSeriesModel(ABC, nn.Module):
    name: str
    family: str
    input_layout: str = "channel_first"
    training_mode: str = "from_scratch"
    pretrained: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config or {}

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape (batch, n_classes)."""
        ...


class OptionalDependencyModel(BaseTimeSeriesModel):
    """Base wrapper for models backed by external libraries or checkpoints."""

    dependency_hint: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        if self.dependency_hint:
            raise ImportError(self.dependency_hint)
