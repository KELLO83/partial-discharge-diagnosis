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
from service.backend.app.models.model_artifacts import (
    AdapterMode,
    MODEL_TASKS,
    ModelAdapterSettings,
    ModelArtifactRecord,
    ModelArtifactRegistry,
    ModelTask,
)

_CHECKPOINT_ADAPTER_TYPES = (
    CheckpointTimeSeriesInferenceAdapter,
    CheckpointVisionInferenceAdapter,
    CheckpointVlmInferenceAdapter,
)


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
class _AdapterSelection:
    mode: AdapterMode
    record: ModelArtifactRecord
    mock_adapter: Any
    checkpoint_factory: Callable[[ModelArtifactRecord], Any]


@dataclass(frozen=True, slots=True)
class _TaskRuntimeSpec:
    mock_attribute: str
    checkpoint_factory: Callable[[ModelArtifactRecord], Any]


@dataclass(frozen=True, slots=True)
class _AdapterBuildRequest:
    mode: AdapterMode
    records: dict[ModelTask, ModelArtifactRecord]
    mock_adapters: MockModelAdapters


@dataclass(frozen=True, slots=True)
class _AdapterInfoLookup:
    mode: AdapterMode
    records: dict[ModelTask, ModelArtifactRecord]
    adapters: dict[ModelTask, Any]


@dataclass(frozen=True, slots=True)
class _AdapterInfoRequest:
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
        return {
            "time_series": self.time_series_info,
            "vision": self.vision_info,
            "vlm": self.vlm_info,
        }[task]


def build_service_model_runtime(
    mock_adapters: MockModelAdapters,
    settings: ModelAdapterSettings | None = None,
) -> ServiceModelRuntime:
    active_settings = settings or ModelAdapterSettings.from_env()
    registry = ModelArtifactRegistry(
        active_settings.artifact_root,
        artifact_overrides=active_settings.artifact_overrides,
    )
    records = registry.all()
    build_request = _AdapterBuildRequest(
        mode=active_settings.mode,
        records=records,
        mock_adapters=mock_adapters,
    )
    adapters = {task: _select_task_adapter(task, build_request) for task in MODEL_TASKS}
    info_lookup = _AdapterInfoLookup(
        mode=active_settings.mode,
        records=records,
        adapters=adapters,
    )

    return ServiceModelRuntime(
        mode=active_settings.mode,
        artifact_root=active_settings.artifact_root,
        time_series_adapter=adapters["time_series"],
        vision_adapter=adapters["vision"],
        vlm_adapter=adapters["vlm"],
        time_series_info=_adapter_info_for_task("time_series", info_lookup),
        vision_info=_adapter_info_for_task("vision", info_lookup),
        vlm_info=_adapter_info_for_task("vlm", info_lookup),
    )


_TASK_RUNTIME_SPECS: dict[ModelTask, _TaskRuntimeSpec] = {
    "time_series": _TaskRuntimeSpec(
        mock_attribute="time_series",
        checkpoint_factory=CheckpointTimeSeriesInferenceAdapter,
    ),
    "vision": _TaskRuntimeSpec(
        mock_attribute="vision",
        checkpoint_factory=CheckpointVisionInferenceAdapter,
    ),
    "vlm": _TaskRuntimeSpec(
        mock_attribute="vlm",
        checkpoint_factory=CheckpointVlmInferenceAdapter,
    ),
}


def _select_task_adapter(task: ModelTask, request: _AdapterBuildRequest) -> Any:
    spec = _TASK_RUNTIME_SPECS[task]
    return _select_adapter(
        _AdapterSelection(
            mode=request.mode,
            record=request.records[task],
            mock_adapter=getattr(request.mock_adapters, spec.mock_attribute),
            checkpoint_factory=spec.checkpoint_factory,
        )
    )


def _adapter_info_for_task(
    task: ModelTask,
    lookup: _AdapterInfoLookup,
) -> ModelAdapterInfo:
    return _adapter_info(
        _AdapterInfoRequest(
            task=task,
            mode=lookup.mode,
            record=lookup.records[task],
            adapter=lookup.adapters[task],
        )
    )


def _select_adapter(selection: _AdapterSelection) -> Any:
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


def _adapter_info(request: _AdapterInfoRequest) -> ModelAdapterInfo:
    is_checkpoint = isinstance(request.adapter, _CHECKPOINT_ADAPTER_TYPES)
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
