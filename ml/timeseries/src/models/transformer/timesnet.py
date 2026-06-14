"""TimesNet wrapper for the official THUML Time-Series-Library implementation."""

from __future__ import annotations

from typing import Any

from ml.timeseries.src.models.base import BaseTimeSeriesModel
from ml.timeseries.src.models.adapters import resize_time_axis_time_first
import torch

from ml.timeseries.src.models.external import clear_external_module_cache, namespace_config, prepend_repo_path
from ml.timeseries.src.schema import DEFAULT_PSEUDO_CHANNELS, N_CLASSES


class TimesNetModel(BaseTimeSeriesModel):
    name = "timesnet"
    family = "transformer"
    input_layout = "channel_first"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        install_hint = (
            "TimesNet requires the official THUML Time-Series-Library. "
            "Clone https://github.com/thuml/Time-Series-Library and set TSLIB_REPO."
        )
        clear_external_module_cache()
        with prepend_repo_path("TSLIB_REPO", install_hint):
            from models.TimesNet import Model
        self.seq_len = int(self.config.get("seq_len", 4096))
        configs = namespace_config(
            self.config,
            task_name="classification",
            seq_len=self.seq_len,
            label_len=0,
            pred_len=0,
            enc_in=DEFAULT_PSEUDO_CHANNELS,
            c_out=DEFAULT_PSEUDO_CHANNELS,
            num_class=N_CLASSES,
            d_model=64,
            d_ff=128,
            e_layers=2,
            top_k=3,
            num_kernels=6,
            dropout=0.1,
            embed="timeF",
            freq="h",
        )
        self.model = Model(configs)

    def forward(self, x):
        x_tf = x.transpose(1, 2)
        x_tf = resize_time_axis_time_first(x_tf, self.seq_len)
        padding_mask = torch.ones(x_tf.shape[0], x_tf.shape[1], device=x_tf.device, dtype=x_tf.dtype)
        return self.model(x_tf, padding_mask, None, None)
