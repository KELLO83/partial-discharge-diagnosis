"""GRU baseline for CSV-only partial discharge classification."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ml.src.models.base import BaseTimeSeriesModel
from ml.src.schema import DEFAULT_PSEUDO_CHANNELS, N_CLASSES


class GRUModel(BaseTimeSeriesModel):
    name = "gru"
    family = "non_transformer"
    input_layout = "time_first"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        cfg = self.config
        hidden_size = int(cfg.get("hidden_size", 128))
        num_layers = int(cfg.get("num_layers", 2))
        dropout = float(cfg.get("dropout", 0.2))
        bidirectional = bool(cfg.get("bidirectional", True))
        self.gru = nn.GRU(
            input_size=int(cfg.get("input_size", DEFAULT_PSEUDO_CHANNELS)),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.use_cudnn = bool(cfg.get("use_cudnn", False))
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, int(cfg.get("n_classes", N_CLASSES))),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cudnn:
            _, hidden = self.gru(x)
        else:
            with torch.backends.cudnn.flags(enabled=False):
                _, hidden = self.gru(x)
        if self.gru.bidirectional:
            features = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            features = hidden[-1]
        return self.head(features)
