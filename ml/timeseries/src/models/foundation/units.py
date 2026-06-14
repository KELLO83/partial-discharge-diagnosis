"""UniTS foundation model wrapper."""

from __future__ import annotations

from typing import Any

import torch

from ml.timeseries.src.models.base import BaseTimeSeriesModel
from ml.timeseries.src.models.adapters import resize_time_axis_time_first
from ml.timeseries.src.models.external import clear_external_module_cache, namespace_config, prepend_repo_path
from ml.timeseries.src.schema import DEFAULT_PSEUDO_CHANNELS, N_CLASSES


class UniTSModel(BaseTimeSeriesModel):
    name = "units"
    family = "foundation"
    input_layout = "time_first"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        install_hint = (
            "UniTS requires the official repo. Clone https://github.com/mims-harvard/UniTS "
            "and set UNITS_REPO."
        )
        clear_external_module_cache()
        with prepend_repo_path("UNITS_REPO", install_hint):
            from models.UniTS import Model
        self.seq_len = int(self.config.get("seq_len", 1024))
        args = namespace_config(
            self.config,
            prompt_num=10,
            d_model=32,
            patch_len=16,
            stride=16,
            dropout=0.1,
            e_layers=1,
            n_heads=4,
        )
        configs_list = [
            (
                "partial_discharge",
                {
                    "dataset": "partial_discharge",
                    "task_name": "classification",
                    "seq_len": self.seq_len,
                    "enc_in": int(self.config.get("input_channels", DEFAULT_PSEUDO_CHANNELS)),
                    "num_class": int(self.config.get("n_classes", N_CLASSES)),
                },
            )
        ]
        self.model = Model(args, configs_list, pretrain=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = resize_time_axis_time_first(x, self.seq_len)
        return self.model(x, None, task_id=0, task_name="classification")
