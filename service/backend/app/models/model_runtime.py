from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from service.backend.app.models.checkpoint_adapters import (
    CheckpointTimeSeriesInferenceAdapter,
    CheckpointVisionInferenceAdapter,
    CheckpointVlmInferenceAdapter,
    ModelAdapterLoadError,
)
from service.backend.app.models.model_artifacts import AdapterMode, ModelAdapterSettings, ModelArtifactRecord, ModelArtifactRegistry, ModelTask


@dataclass(frozen=True, slots=True)
class ModelAdapterInfo:
    task: ModelTask
    adapter_kind: str
    model_name: str
    model_version: str
    ready: bool
    manifest_path: str | None = None
    checkpoint_path: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MockModelAdapters:
    time_series: Any
    vision: Any
    vlm: Any


@dataclass(frozen=True, slots=True)
class AdapterSelection:
    mode: AdapterMode
    record: ModelArtifactRecord
    mock_adapter: Any
    checkpoint_factory: Callable[[ModelArtifactRecord], Any]


@dataclass(frozen=True, slots=True)
class AdapterInfoRequest:
    task: ModelTask
    mode: AdapterMode
    record: ModelArtifactRecord
    adapter: Any


@dataclass(frozen=True, slots=True)
class ServiceModelRuntime:
    mode: AdapterMode
    artifact_root: Path
    time_series_adapter: Any
    vision_adapter: Any
    vlm_adapter: Any
    time_series_info: ModelAdapterInfo
    vision_info: ModelAdapterInfo
    vlm_info: ModelAdapterInfo

    def info_for(self, task: ModelTask) -> ModelAdapterInfo:
        if task == "time_series":
            return self.time_series_info
        if task == "vision":
            return self.vision_info
        return self.vlm_info


def build_service_model_runtime(
    mock_adapters: MockModelAdapters,
    settings: ModelAdapterSettings | None = None,
) -> ServiceModelRuntime:
    active_settings = settings or ModelAdapterSettings.from_env()
    registry = ModelArtifactRegistry(active_settings.artifact_root)
    time_series_record = registry.get("time_series")
    vision_record = registry.get("vision")
    vlm_record = registry.get("vlm")

    time_series_adapter = _select_adapter(
        AdapterSelection(
            mode=active_settings.mode,
            record=time_series_record,
            mock_adapter=mock_adapters.time_series,
            checkpoint_factory=lambda record: CheckpointTimeSeriesInferenceAdapter(record),
        )
    )
    vision_adapter = _select_adapter(
        AdapterSelection(
            mode=active_settings.mode,
            record=vision_record,
            mock_adapter=mock_adapters.vision,
            checkpoint_factory=lambda record: CheckpointVisionInferenceAdapter(record),
        )
    )
    vlm_adapter = _select_adapter(
        AdapterSelection(
            mode=active_settings.mode,
            record=vlm_record,
            mock_adapter=mock_adapters.vlm,
            checkpoint_factory=lambda record: CheckpointVlmInferenceAdapter(record),
        )
    )

    return ServiceModelRuntime(
        mode=active_settings.mode,
        artifact_root=active_settings.artifact_root,
        time_series_adapter=time_series_adapter,
        vision_adapter=vision_adapter,
        vlm_adapter=vlm_adapter,
        time_series_info=_adapter_info(
            AdapterInfoRequest(
                task="time_series",
                mode=active_settings.mode,
                record=time_series_record,
                adapter=time_series_adapter,
            )
        ),
        vision_info=_adapter_info(
            AdapterInfoRequest(
                task="vision",
                mode=active_settings.mode,
                record=vision_record,
                adapter=vision_adapter,
            )
        ),
        vlm_info=_adapter_info(
            AdapterInfoRequest(
                task="vlm",
                mode=active_settings.mode,
                record=vlm_record,
                adapter=vlm_adapter,
            )
        ),
    )


def _select_adapter(selection: AdapterSelection) -> Any:
    if selection.mode == "mock":
        return selection.mock_adapter
    if selection.mode == "auto" and not selection.record.ready:
        return selection.mock_adapter
    try:
        return selection.checkpoint_factory(selection.record)
    except ModelAdapterLoadError:
        if selection.mode == "auto":
            return selection.mock_adapter
        raise


def _adapter_info(request: AdapterInfoRequest) -> ModelAdapterInfo:
    is_checkpoint = isinstance(
        request.adapter,
        (CheckpointTimeSeriesInferenceAdapter, CheckpointVisionInferenceAdapter, CheckpointVlmInferenceAdapter),
    )
    adapter_kind = "checkpoint" if is_checkpoint else "mock"
    error = None
    if request.mode == "checkpoint" and not request.record.ready:
        error = request.record.error
    elif request.mode == "auto" and not is_checkpoint:
        error = request.record.error
    return ModelAdapterInfo(
        task=request.task,
        adapter_kind=adapter_kind,
        model_name=str(getattr(request.adapter, "model_name", request.record.model_name)),
        model_version=str(getattr(request.adapter, "model_version", request.record.model_version)),
        ready=is_checkpoint or adapter_kind == "mock",
        manifest_path=str(request.record.manifest_path) if request.record.manifest_path.exists() else None,
        checkpoint_path=str(request.record.checkpoint_path) if request.record.checkpoint_path is not None else None,
        error=error,
    )
