"""MiniROCKET wrapper.

MiniROCKET is intentionally kept as an optional external-library wrapper because
its reference implementations are not part of PyTorch.
"""

from __future__ import annotations

from typing import Any

import torch

from ml.src.models.base import BaseTimeSeriesModel
from ml.src.models.external import require_module


class MiniROCKETModel(BaseTimeSeriesModel):
    name = "minirocket"
    family = "non_transformer"
    input_layout = "channel_first"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        require_module(
            "sktime.transformations.panel.rocket",
            "MiniROCKET requires the verified sktime implementation. Install sktime and use "
            "MiniRocketMultivariate/MiniRocketMultivariateVariable through a sklearn-style runner.",
        )
        raise ImportError("MiniROCKET is an official sktime sklearn-style pipeline. Use ml/scripts/run_minirocket.py.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("MiniROCKET is a sklearn-style pipeline, not a torch forward model.")
