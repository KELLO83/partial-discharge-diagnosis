"""ModernTCN wrapper for the official classification implementation."""

from __future__ import annotations

from typing import Any

import torch

from ml.src.models.adapters import resize_time_axis_time_first
from ml.src.models.base import BaseTimeSeriesModel
from ml.src.models.external import clear_external_module_cache, namespace_config, prepend_repo_path
from ml.src.schema import DEFAULT_PSEUDO_CHANNELS, N_CLASSES


class ModernTCNModel(BaseTimeSeriesModel):
    name = "moderntcn"
    family = "non_transformer"
    input_layout = "time_first"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        install_hint = (
            "ModernTCN requires the official repo. Clone https://github.com/luodhhh/ModernTCN "
            "and set MODERNTCN_REPO to the ModernTCN-classification directory."
        )
        clear_external_module_cache()
        with prepend_repo_path("MODERNTCN_REPO", install_hint):
            from models.ModernTCN import Model

        self.seq_len = int(self.config.get("seq_len", 4096))
        configs = namespace_config(
            self.config,
            task_name="classification",
            seq_len=self.seq_len,
            label_len=0,
            pred_len=0,
            enc_in=int(self.config.get("input_channels", DEFAULT_PSEUDO_CHANNELS)),
            num_class=int(self.config.get("n_classes", N_CLASSES)),
            patch_size=16,
            patch_stride=8,
            stem_ratio=6,
            downsample_ratio=2,
            ffn_ratio=1,
            num_blocks=[1, 1],
            large_size=[13, 13],
            small_size=[5, 5],
            dims=[32, 64],
            dw_dims=[32, 64],
            dropout=0.1,
            head_dropout=0.0,
            class_dropout=0.1,
            use_multi_scale=False,
            small_kernel_merged=False,
            revin=False,
            affine=True,
            subtract_last=False,
            freq=None,
            individual=False,
            kernel_size=25,
            decomposition=False,
        )
        self.model = Model(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = resize_time_axis_time_first(x, self.seq_len)
        return self.model(x, None, None, None)
