"""TS2Vec representation learning wrapper."""

from __future__ import annotations

from typing import Any

import torch

from ml.timeseries.src.models.base import BaseTimeSeriesModel
from ml.timeseries.src.models.external import clear_external_module_cache, prepend_repo_path


class TS2VecModel(BaseTimeSeriesModel):
    name = "ts2vec"
    family = "representation"
    input_layout = "time_first"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        install_hint = (
            "TS2Vec requires the official repo. Clone https://github.com/yuezhihan/ts2vec "
            "and set TS2VEC_REPO."
        )
        clear_external_module_cache()
        with prepend_repo_path("TS2VEC_REPO", install_hint):
            from ts2vec import TS2Vec
        self.encoder_cls = TS2Vec
        raise ImportError(
            "TS2Vec official encoder is available, but this project still needs a fitted encoder checkpoint "
            "and classifier head adapter before running it through the common torch runner."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
