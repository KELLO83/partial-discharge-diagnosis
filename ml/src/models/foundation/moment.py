"""MOMENT foundation model wrapper."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ml.src.models.base import BaseTimeSeriesModel
from ml.src.models.adapters import resize_time_axis_channel_first
from ml.src.models.external import require_module
from ml.src.schema import DEFAULT_PSEUDO_CHANNELS, N_CLASSES


class MOMENTModel(BaseTimeSeriesModel):
    name = "moment"
    family = "foundation"
    input_layout = "channel_first"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        cfg = self.config
        module = require_module(
            "momentfm",
            "MOMENT requires the official momentfm package. Install momentfm or "
            "git+https://github.com/moment-timeseries-foundation-model/moment.git.",
        )
        MOMENTPipeline = module.MOMENTPipeline
        checkpoint = str(cfg.get("checkpoint", "AutonLab/MOMENT-1-small"))
        self.seq_len = int(cfg.get("seq_len", 512))
        self.backbone = MOMENTPipeline.from_pretrained(
            checkpoint,
            model_kwargs={
                "task_name": "embedding",
                "n_channels": int(cfg.get("input_channels", DEFAULT_PSEUDO_CHANNELS)),
            },
        )
        self.backbone.init()
        self.head = nn.Linear(int(cfg.get("embedding_dim", 512)), int(cfg.get("n_classes", N_CLASSES)))
        self.freeze_backbone = bool(cfg.get("freeze_backbone", True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = resize_time_axis_channel_first(x, self.seq_len)
        with torch.set_grad_enabled(not self.freeze_backbone):
            output = self.backbone(x_enc=x)
        embeddings = output.embeddings
        if embeddings.ndim == 3:
            embeddings = embeddings.mean(dim=1)
        return self.head(embeddings)
