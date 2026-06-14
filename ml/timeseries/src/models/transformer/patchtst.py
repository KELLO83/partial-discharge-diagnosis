"""PatchTST wrapper backed by Hugging Face Transformers."""

from __future__ import annotations

from typing import Any

import torch

from ml.timeseries.src.models.base import BaseTimeSeriesModel
from ml.timeseries.src.models.external import require_module
from ml.timeseries.src.schema import DEFAULT_PSEUDO_CHANNELS, N_CLASSES


class PatchTSTModel(BaseTimeSeriesModel):
    name = "patchtst"
    family = "transformer"
    input_layout = "time_first"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        cfg = self.config
        module = require_module(
            "transformers",
            "Install Hugging Face Transformers with PatchTST support: pip install transformers",
        )
        PatchTSTConfig = module.PatchTSTConfig
        PatchTSTForClassification = module.PatchTSTForClassification

        hf_config = PatchTSTConfig(
            num_input_channels=int(cfg.get("num_input_channels", DEFAULT_PSEUDO_CHANNELS)),
            num_targets=int(cfg.get("n_classes", N_CLASSES)),
            context_length=int(cfg.get("context_length", 7680)),
            patch_length=int(cfg.get("patch_length", 64)),
            patch_stride=int(cfg.get("patch_stride", 32)),
            d_model=int(cfg.get("d_model", 128)),
            num_hidden_layers=int(cfg.get("num_hidden_layers", 3)),
            num_attention_heads=int(cfg.get("num_attention_heads", 4)),
        )
        self.model = PatchTSTForClassification(hf_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(past_values=x)
        for field in ("logits", "prediction_logits", "prediction_outputs"):
            value = getattr(output, field, None)
            if value is not None:
                return value
        if isinstance(output, tuple):
            return output[0]
        raise AttributeError(f"Unsupported PatchTST output fields: {output}")
