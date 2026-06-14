"""GPT4TS / One-Fits-All wrapper."""

from __future__ import annotations

from typing import Any

import torch
import pandas as pd

from ml.timeseries.src.models.base import BaseTimeSeriesModel
from ml.timeseries.src.models.adapters import resize_time_axis_time_first
from ml.timeseries.src.models.external import clear_external_module_cache, namespace_config, prepend_repo_path
from ml.timeseries.src.schema import DEFAULT_PSEUDO_CHANNELS, DEFAULT_SEQUENCE_LENGTH, LABEL_ID_TO_NAME


class GPT4TSModel(BaseTimeSeriesModel):
    name = "gpt4ts"
    family = "foundation"
    input_layout = "time_first"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        install_hint = (
            "GPT4TS / One-Fits-All requires the official repo. Clone "
            "https://github.com/DAMO-DI-ML/One_Fits_All and set ONE_FITS_ALL_REPO."
        )
        clear_external_module_cache()
        with prepend_repo_path("ONE_FITS_ALL_REPO", install_hint):
            import sys
            from pathlib import Path

            classification_src = Path("external/One_Fits_All/Classification/src").resolve()
            sys.path.insert(0, str(classification_src))
            try:
                from models.gpt4ts import gpt4ts
            finally:
                try:
                    sys.path.remove(str(classification_src))
                except ValueError:
                    pass
        self.seq_len = int(self.config.get("seq_len", 1024))
        data = namespace_config(
            max_seq_len=self.seq_len,
            feature_df=pd.DataFrame(columns=[f"ch_{idx}" for idx in range(DEFAULT_PSEUDO_CHANNELS)]),
            class_names=[LABEL_ID_TO_NAME[idx] for idx in sorted(LABEL_ID_TO_NAME)],
        )
        config = {
            "patch_size": int(self.config.get("patch_size", 16)),
            "stride": int(self.config.get("stride", 8)),
            "d_model": int(self.config.get("d_model", 768)),
            "dropout": float(self.config.get("dropout", 0.1)),
        }
        self.model = gpt4ts(config, data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = resize_time_axis_time_first(x, self.seq_len)
        return self.model(x, None)
