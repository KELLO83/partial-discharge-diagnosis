"""ResNet1D wrapper using the verified tsai implementation."""

from __future__ import annotations

from typing import Any

from ml.timeseries.src.models.base import BaseTimeSeriesModel
from ml.timeseries.src.models.external import require_module
from ml.timeseries.src.schema import DEFAULT_PSEUDO_CHANNELS, N_CLASSES


class ResNet1DModel(BaseTimeSeriesModel):
    name = "resnet1d"
    family = "non_transformer"
    input_layout = "channel_first"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        cfg = self.config
        module = require_module(
            "tsai.models.ResNet",
            "ResNet1D requires the verified tsai implementation. Install tsai before running this model.",
        )
        self.model = module.ResNet(
            c_in=int(cfg.get("input_channels", DEFAULT_PSEUDO_CHANNELS)),
            c_out=int(cfg.get("n_classes", N_CLASSES)),
        )

    def forward(self, x):
        return self.model(x)
