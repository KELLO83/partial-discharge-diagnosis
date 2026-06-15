"""Model registry for partial discharge time-series experiments."""

from __future__ import annotations

from typing import Any

from ml.timeseries.src.models.base import BaseTimeSeriesModel
from ml.timeseries.src.models.foundation.gpt4ts import GPT4TSModel
from ml.timeseries.src.models.foundation.moment import MOMENTModel
from ml.timeseries.src.models.foundation.units import UniTSModel
from ml.timeseries.src.models.non_transformer.gru import GRUModel
from ml.timeseries.src.models.non_transformer.inception_time import InceptionTimeModel
from ml.timeseries.src.models.non_transformer.inception_time_small import InceptionTimeSmallModel
from ml.timeseries.src.models.non_transformer.minirocket import MiniROCKETModel
from ml.timeseries.src.models.non_transformer.moderntcn import ModernTCNModel
from ml.timeseries.src.models.non_transformer.resnet1d import ResNet1DModel
from ml.timeseries.src.models.non_transformer.tcn import TCNModel
from ml.timeseries.src.models.representation.ts2vec import TS2VecModel
from ml.timeseries.src.models.transformer.itransformer import ITransformerModel
from ml.timeseries.src.models.transformer.patchtst import PatchTSTModel
from ml.timeseries.src.models.transformer.timemixer import TimeMixerModel
from ml.timeseries.src.models.transformer.timesnet import TimesNetModel


MODEL_REGISTRY: dict[str, type[BaseTimeSeriesModel]] = {
    "gru": GRUModel,
    "tcn": TCNModel,
    "inception_time": InceptionTimeModel,
    "inception_time_small": InceptionTimeSmallModel,
    "resnet1d": ResNet1DModel,
    "minirocket": MiniROCKETModel,
    "moderntcn": ModernTCNModel,
    "patchtst": PatchTSTModel,
    "itransformer": ITransformerModel,
    "timesnet": TimesNetModel,
    "timemixer": TimeMixerModel,
    "moment": MOMENTModel,
    "units": UniTSModel,
    "gpt4ts": GPT4TSModel,
    "ts2vec": TS2VecModel,
}


def create_model(model_name: str, params: dict[str, Any] | None = None) -> BaseTimeSeriesModel:
    try:
        model_cls = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model: {model_name}. Supported: {supported}") from exc
    return model_cls(config=params)
