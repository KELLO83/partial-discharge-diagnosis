"""TCN wrapper using a verified external implementation when available."""

from __future__ import annotations

from typing import Any

from ml.src.models.base import BaseTimeSeriesModel
from ml.src.models.external import require_module
from ml.src.schema import DEFAULT_PSEUDO_CHANNELS, N_CLASSES


class TCNModel(BaseTimeSeriesModel):
    name = "tcn"
    family = "non_transformer"
    input_layout = "channel_first"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        cfg = self.config
        module = require_module(
            "tsai.models.TCN",
            "TCN requires the verified tsai implementation. Install a Python-compatible tsai package "
            "or connect the official locuslab/TCN repo through a dedicated adapter.",
        )
        self.model = module.TCN(
            c_in=int(cfg.get("input_channels", DEFAULT_PSEUDO_CHANNELS)),
            c_out=int(cfg.get("n_classes", N_CLASSES)),
            layers=list(cfg.get("layers", [64, 128, 128])),
            ks=int(cfg.get("kernel_size", 7)),
        )

    def forward(self, x):
        return self.model(x)
