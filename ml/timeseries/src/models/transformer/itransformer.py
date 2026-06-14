"""iTransformer wrapper for the official THUML implementation."""

from __future__ import annotations

from typing import Any

from ml.timeseries.src.models.base import BaseTimeSeriesModel
from ml.timeseries.src.models.external import clear_external_module_cache, namespace_config, prepend_repo_path
from ml.timeseries.src.schema import DEFAULT_PSEUDO_CHANNELS, DEFAULT_SEQUENCE_LENGTH, N_CLASSES


class ITransformerModel(BaseTimeSeriesModel):
    name = "itransformer"
    family = "transformer"
    input_layout = "time_first"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        install_hint = (
            "iTransformer requires the official THUML repo. Clone https://github.com/thuml/iTransformer "
            "or https://github.com/thuml/Time-Series-Library and set ITRANSFORMER_REPO or TSLIB_REPO."
        )
        try:
            clear_external_module_cache()
            with prepend_repo_path("TSLIB_REPO", install_hint):
                from models.iTransformer import Model
        except ImportError:
            clear_external_module_cache()
            with prepend_repo_path("ITRANSFORMER_REPO", install_hint):
                from model.iTransformer import Model
        configs = namespace_config(
            self.config,
            task_name="classification",
            seq_len=DEFAULT_SEQUENCE_LENGTH,
            label_len=0,
            pred_len=0,
            enc_in=DEFAULT_PSEUDO_CHANNELS,
            c_out=DEFAULT_PSEUDO_CHANNELS,
            num_class=N_CLASSES,
            d_model=128,
            n_heads=4,
            e_layers=3,
            d_ff=256,
            dropout=0.1,
            activation="gelu",
            embed="timeF",
            freq="h",
            factor=1,
            output_attention=False,
            use_norm=1,
            class_strategy="projection",
        )
        self.model = Model(configs)

    def forward(self, x):
        return self.model(x, None, None, None)
