from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from service.backend.app.application.contracts import (
    TimeSeriesInferenceAdapter,
    TimeSeriesToolInput,
    VisionInferenceAdapter,
    VisionToolInput,
    VlmInferenceAdapter,
    VlmToolInput,
)
from service.backend.app.domain.fusion import time_series_evidence, vision_evidence, vlm_evidence
from service.backend.app.domain.policy import label_name, recommended_action, risk_level
from service.backend.app.models.model_artifacts import ModelArtifactManifest, ModelArtifactRecord
from service.backend.app.schemas import TimeSeriesResult, VisionResult, VlmResult

TIME_SERIES_PREDICTION_METHODS = ("predict_csv", "predict")
VISION_PREDICTION_METHODS = ("predict_image", "predict")
VLM_PREDICTION_METHODS = ("generate_report", "predict")
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0


class ModelAdapterLoadError(RuntimeError):
    pass


class ModelInferenceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelRuntimeContext:
    manifest: ModelArtifactManifest
    manifest_path: Path
    checkpoint_path: Path | None
    preprocessor_path: Path | None


class CheckpointTimeSeriesInferenceAdapter(TimeSeriesInferenceAdapter):
    def __init__(self, record: ModelArtifactRecord, backend: object | None = None) -> None:
        self.record = _require_ready_record(record)
        manifest = _require_manifest(self.record)
        self.model_name = manifest.model_name
        self.model_version = manifest.model_version
        self._backend = backend

    def run(self, tool_input: TimeSeriesToolInput) -> TimeSeriesResult:
        raw = _predict(self._load_backend(), TIME_SERIES_PREDICTION_METHODS, tool_input)
        result = _timeseries_result(raw, self.model_name, self.model_version)
        return result.model_copy(update={"standard_evidence": time_series_evidence(result)})

    def _load_backend(self) -> object:
        if self._backend is None:
            self._backend = load_backend(self.record)
        return self._backend


class CheckpointVisionInferenceAdapter(VisionInferenceAdapter):
    def __init__(self, record: ModelArtifactRecord, backend: object | None = None) -> None:
        self.record = _require_ready_record(record)
        manifest = _require_manifest(self.record)
        self.model_name = manifest.model_name
        self.model_version = manifest.model_version
        self._backend = backend

    def run(self, tool_input: VisionToolInput) -> VisionResult:
        raw = _predict(self._load_backend(), VISION_PREDICTION_METHODS, tool_input)
        result = _vision_result(raw, self.model_name, self.model_version)
        return result.model_copy(update={"standard_evidence": vision_evidence(result)})

    def _load_backend(self) -> object:
        if self._backend is None:
            self._backend = load_backend(self.record)
        return self._backend


class CheckpointVlmInferenceAdapter(VlmInferenceAdapter):
    def __init__(self, record: ModelArtifactRecord, backend: object | None = None) -> None:
        self.record = _require_ready_record(record)
        manifest = _require_manifest(self.record)
        self.model_name = manifest.model_name
        self.model_version = manifest.model_version
        self._backend = backend

    def run(self, tool_input: VlmToolInput) -> VlmResult:
        raw = _predict(self._load_backend(), VLM_PREDICTION_METHODS, tool_input)
        result = _vlm_result(raw, self.model_name, self.model_version)
        return result.model_copy(update={"standard_evidence": vlm_evidence(result)})

    def _load_backend(self) -> object:
        if self._backend is None:
            self._backend = load_backend(self.record)
        return self._backend


def load_backend(record: ModelArtifactRecord) -> object:
    manifest = record.manifest
    if manifest is None:
        raise ModelAdapterLoadError(record.error or "model manifest is not configured")
    module_name, factory_name = _split_entrypoint(manifest.entrypoint)
    try:
        module = import_module(module_name)
        factory = getattr(module, factory_name)
    except (ImportError, AttributeError) as exc:
        raise ModelAdapterLoadError(f"cannot load model entrypoint {manifest.entrypoint}: {exc}") from exc
    return factory(
        ModelRuntimeContext(
            manifest=manifest,
            manifest_path=record.manifest_path,
            checkpoint_path=record.checkpoint_path,
            preprocessor_path=record.preprocessor_path,
        )
    )


def _require_ready_record(record: ModelArtifactRecord) -> ModelArtifactRecord:
    if not record.ready or record.manifest is None:
        raise ModelAdapterLoadError(record.error or f"{record.task} model artifact is not ready")
    return record


def _require_manifest(record: ModelArtifactRecord) -> ModelArtifactManifest:
    if record.manifest is None:
        raise ModelAdapterLoadError(record.error or f"{record.task} model manifest is not ready")
    return record.manifest


def _split_entrypoint(entrypoint: str) -> tuple[str, str]:
    if ":" not in entrypoint:
        raise ModelAdapterLoadError("model entrypoint must use module:function format")
    module_name, factory_name = entrypoint.split(":", 1)
    if module_name.strip() == "" or factory_name.strip() == "":
        raise ModelAdapterLoadError("model entrypoint must use module:function format")
    return module_name, factory_name


def _predict(backend: object, method_names: tuple[str, ...], tool_input: object) -> dict[str, object]:
    for method_name in method_names:
        method = getattr(backend, method_name, None)
        if callable(method):
            output = method(tool_input)
            if isinstance(output, dict):
                return output
            if hasattr(output, "model_dump"):
                return output.model_dump()
            raise ModelInferenceError(f"{method_name} must return dict or pydantic model")
    raise ModelInferenceError(f"backend must implement one of: {', '.join(method_names)}")


def _timeseries_result(raw: dict[str, object], model_name: str, model_version: str) -> TimeSeriesResult:
    label_id = _required_int(raw, "label_id")
    confidence = _required_float(raw, "confidence")
    return TimeSeriesResult(
        model_name=str(raw.get("model_name") or model_name),
        model_version=str(raw.get("model_version") or model_version),
        label_id=label_id,
        label_name=str(raw.get("label_name") or label_name(label_id)),
        confidence=confidence,
        probabilities=_probabilities(raw.get("probabilities"), label_id, confidence),
        features=_float_map(raw.get("features")),
    )


def _vision_result(raw: dict[str, object], model_name: str, model_version: str) -> VisionResult:
    label_id = _required_int(raw, "label_id")
    confidence = _required_float(raw, "confidence")
    return VisionResult(
        model_name=str(raw.get("model_name") or model_name),
        model_version=str(raw.get("model_version") or model_version),
        label_id=label_id,
        label_name=str(raw.get("label_name") or label_name(label_id)),
        confidence=confidence,
        probabilities=_probabilities(raw.get("probabilities"), label_id, confidence),
        evidence=_vision_evidence(raw.get("evidence")),
    )


def _vlm_result(raw: dict[str, object], model_name: str, model_version: str) -> VlmResult:
    label_id = _required_int(raw, "label_id")
    diagnosis = str(raw.get("diagnosis") or raw.get("label_name") or label_name(label_id))
    return VlmResult(
        model_name=str(raw.get("model_name") or model_name),
        model_version=str(raw.get("model_version") or model_version),
        label_id=label_id,
        diagnosis=diagnosis,
        risk_level=str(raw.get("risk_level") or risk_level(label_id)),
        confidence=_required_float(raw, "confidence"),
        reason=str(raw.get("reason") or f"{diagnosis} 모델 판단 결과입니다."),
        recommended_action=str(raw.get("recommended_action") or recommended_action(label_id)),
    )


def _required_int(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool):
        raise ModelInferenceError(f"{key} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ModelInferenceError(f"{key} must be an integer")


def _required_float(raw: dict[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return _clamp_confidence(float(value))
    raise ModelInferenceError(f"{key} must be a float in [0, 1]")


def _probabilities(value: object, label_id: int, confidence: float) -> dict[str, float]:
    if isinstance(value, dict):
        probabilities: dict[str, float] = {}
        for key, item in value.items():
            if isinstance(item, int | float) and not isinstance(item, bool):
                probabilities[str(key)] = _clamp_confidence(float(item))
        if probabilities:
            return probabilities
    return {str(label_id): confidence}


def _clamp_confidence(value: float) -> float:
    return max(MIN_CONFIDENCE, min(value, MAX_CONFIDENCE))


def _float_map(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(item, int | float) and not isinstance(item, bool)
    }


def _vision_evidence(value: object) -> dict[str, float | str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, str) or (isinstance(item, int | float) and not isinstance(item, bool))
    }
