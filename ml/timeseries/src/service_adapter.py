from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ml.timeseries.src.data.loader import read_timeseries_csv
from ml.timeseries.src.features.timeseries_summary import summarize_signal
from ml.timeseries.src.models.registry import create_model
from ml.timeseries.src.schema import LABEL_ID_TO_NAME


class TimeSeriesServiceAdapter:
    def __init__(self, context: object) -> None:
        self.context = context
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._checkpoint: dict[str, object] | None = None
        self._preprocessor: dict[str, object] | None = None

    def predict_csv(self, tool_input: object) -> dict[str, object]:
        checkpoint = self._load_checkpoint()
        preprocessor = self._load_preprocessor()
        model = self._load_model(checkpoint)

        csv_path = Path(getattr(tool_input, "csv_path"))
        signal = read_timeseries_csv(csv_path)
        if _model_layout(checkpoint, preprocessor) == "time_first":
            signal = signal.T
        signal_tensor = torch.from_numpy(np.asarray(signal, dtype=np.float32)).unsqueeze(0).to(self.device)
        signal_tensor = _normalize_signal(signal_tensor, preprocessor)

        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(signal_tensor), dim=-1)[0].cpu().numpy()

        label_id = int(np.argmax(probabilities))
        summary = summarize_signal(signal)
        return {
            "model_name": str(checkpoint.get("model_name", "pd_timeseries_model")),
            "model_version": _model_version(self.context, checkpoint),
            "label_id": label_id,
            "label_name": LABEL_ID_TO_NAME.get(label_id, "unknown"),
            "confidence": float(probabilities[label_id]),
            "probabilities": {str(index): float(value) for index, value in enumerate(probabilities)},
            "features": _round_dict(summary.to_float_dict()),
        }

    def _load_model(self, checkpoint: dict[str, object]) -> torch.nn.Module:
        if self._model is None:
            model_name = str(checkpoint.get("model_name", "gru"))
            model = create_model(model_name, params=_model_params(checkpoint))
            state = checkpoint.get("model_state_dict")
            if isinstance(state, dict):
                model.load_state_dict(state)
            self._model = model.to(self.device)
        return self._model

    def _load_checkpoint(self) -> dict[str, object]:
        if self._checkpoint is None:
            checkpoint_path = _checkpoint_path(self.context)
            self._checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if not isinstance(self._checkpoint, dict):
                raise RuntimeError("time series checkpoint must be a serialized dictionary.")
        return self._checkpoint

    def _load_preprocessor(self) -> dict[str, object]:
        if self._preprocessor is None:
            preprocessor_path = getattr(self.context, "preprocessor_path", None)
            if preprocessor_path is None:
                self._preprocessor = {}
            else:
                path = Path(preprocessor_path)
                if not path.exists():
                    raise RuntimeError(f"time series preprocessor path does not exist: {preprocessor_path}")
                self._preprocessor = json.loads(path.read_text(encoding="utf-8"))
        return self._preprocessor


def load_adapter(context: object) -> TimeSeriesServiceAdapter:
    return TimeSeriesServiceAdapter(context)


def _checkpoint_path(context: object) -> Path:
    checkpoint_path = getattr(context, "checkpoint_path", None)
    if checkpoint_path is None:
        raise RuntimeError("time series checkpoint path is not configured.")
    return Path(checkpoint_path)


def _model_layout(checkpoint: dict[str, object], preprocessor: dict[str, object]) -> str:
    explicit_layout = _first_str(checkpoint.get("input_layout"))
    explicit_layout = explicit_layout or _first_str(preprocessor.get("input_layout"))
    if explicit_layout in {"channel_first", "time_first"}:
        return explicit_layout
    return "channel_first"


def _model_params(checkpoint: dict[str, object]) -> dict[str, object]:
    params = checkpoint.get("model_params")
    if not isinstance(params, dict):
        return {}
    return {str(key): value for key, value in params.items()}


def _normalize_signal(signal: torch.Tensor, preprocessor: dict[str, object]) -> torch.Tensor:
    if str(preprocessor.get("normalize", "true")).lower() in {"false", "0", "no"}:
        return signal
    if signal.ndim != 3:
        return signal
    mean = signal.mean(dim=(-2, -1), keepdim=True)
    std = signal.std(dim=(-2, -1), keepdim=True)
    return (signal - mean) / torch.clamp(std, min=1e-6)


def _round_dict(values: dict[str, float]) -> dict[str, float]:
    return {str(key): round(float(value), 8) for key, value in values.items()}


def _first_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return str(value)
    return None


def _model_version(context: object, checkpoint: dict[str, object]) -> str:
    manifest = getattr(context, "manifest", None)
    manifest_version = getattr(manifest, "model_version", None) if manifest is not None else None
    if manifest_version is not None:
        return str(manifest_version)
    return str(checkpoint.get("model_version", "unknown"))
