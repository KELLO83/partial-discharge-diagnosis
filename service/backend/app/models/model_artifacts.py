from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models"
MANIFEST_FILENAME = "model_manifest.json"

ModelTask = Literal["time_series", "vision", "vlm"]
AdapterMode = Literal["mock", "checkpoint", "auto"]


class ModelInputSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    modality: str
    schema_version: str = "1.0"
    shape: list[int | str] = Field(default_factory=list)
    dtype: str | None = None
    notes: str | None = None


class ModelOutputSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    schema_version: str = "1.0"
    required_fields: list[str] = Field(default_factory=list)
    notes: str | None = None


class ModelArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    task: ModelTask
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    framework: str = Field(default="python")
    entrypoint: str = Field(
        min_length=1,
        description="Python entrypoint in module:function form. The function must return an inference backend.",
    )
    checkpoint_path: str | None = None
    preprocessor_path: str | None = None
    label_map: dict[str, str] = Field(default_factory=dict)
    input_spec: ModelInputSpec
    output_spec: ModelOutputSpec
    thresholds: dict[str, float] = Field(default_factory=dict)
    runtime: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelArtifactRecord:
    task: ModelTask
    manifest_path: Path
    manifest: ModelArtifactManifest | None
    checkpoint_path: Path | None
    preprocessor_path: Path | None
    ready: bool
    error: str | None = None

    @property
    def model_name(self) -> str:
        return self.manifest.model_name if self.manifest is not None else "not_configured"

    @property
    def model_version(self) -> str:
        return self.manifest.model_version if self.manifest is not None else "not_configured"


@dataclass(frozen=True, slots=True)
class ModelAdapterSettings:
    mode: AdapterMode
    artifact_root: Path

    @classmethod
    def from_env(cls) -> "ModelAdapterSettings":
        return cls(
            mode=_env_mode("MODEL_ADAPTER_MODE", "mock"),
            artifact_root=_artifact_root(os.getenv("MODEL_ARTIFACT_ROOT")),
        )


class ModelArtifactRegistry:
    def __init__(self, artifact_root: Path = DEFAULT_MODEL_ARTIFACT_ROOT) -> None:
        self.artifact_root = artifact_root

    def get(self, task: ModelTask) -> ModelArtifactRecord:
        manifest_path = self.artifact_root / task / MANIFEST_FILENAME
        if not manifest_path.exists():
            return ModelArtifactRecord(
                task=task,
                manifest_path=manifest_path,
                manifest=None,
                checkpoint_path=None,
                preprocessor_path=None,
                ready=False,
                error=f"manifest not found: {manifest_path}",
            )
        try:
            manifest = ModelArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            return ModelArtifactRecord(
                task=task,
                manifest_path=manifest_path,
                manifest=None,
                checkpoint_path=None,
                preprocessor_path=None,
                ready=False,
                error=str(exc),
            )
        if manifest.task != task:
            return ModelArtifactRecord(
                task=task,
                manifest_path=manifest_path,
                manifest=manifest,
                checkpoint_path=None,
                preprocessor_path=None,
                ready=False,
                error=f"manifest task mismatch: expected {task}, got {manifest.task}",
            )
        checkpoint_path = _resolve_optional_path(manifest_path, manifest.checkpoint_path)
        preprocessor_path = _resolve_optional_path(manifest_path, manifest.preprocessor_path)
        missing_path = _first_missing_path(checkpoint_path, preprocessor_path)
        return ModelArtifactRecord(
            task=task,
            manifest_path=manifest_path,
            manifest=manifest,
            checkpoint_path=checkpoint_path,
            preprocessor_path=preprocessor_path,
            ready=missing_path is None,
            error=None if missing_path is None else f"artifact path not found: {missing_path}",
        )

    def all(self) -> dict[ModelTask, ModelArtifactRecord]:
        return {
            "time_series": self.get("time_series"),
            "vision": self.get("vision"),
            "vlm": self.get("vlm"),
        }


def _env_mode(name: str, fallback: AdapterMode) -> AdapterMode:
    value = os.getenv(name, fallback).strip().lower()
    if value in {"mock", "checkpoint", "auto"}:
        return value  # type: ignore[return-value]
    return fallback


def _artifact_root(value: str | None) -> Path:
    if value is None or value.strip() == "":
        return DEFAULT_MODEL_ARTIFACT_ROOT
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_optional_path(manifest_path: Path, value: str | None) -> Path | None:
    if value is None or value.strip() == "":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _first_missing_path(*paths: Path | None) -> Path | None:
    for path in paths:
        if path is not None and not path.exists():
            return path
    return None
