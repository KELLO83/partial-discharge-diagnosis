from __future__ import annotations

import json
from pathlib import Path

from service.backend.app.models.checkpoint_adapters import (
    CheckpointTimeSeriesInferenceAdapter,
    CheckpointVisionInferenceAdapter,
    CheckpointVlmInferenceAdapter,
)
from service.backend.app.models.model_artifacts import ModelAdapterSettings, ModelArtifactRegistry, ModelTask
from service.backend.app.models.model_runtime import MockModelAdapters, build_service_model_runtime
from service.backend.app.application.contracts import TimeSeriesToolInput, VisionToolInput, VlmToolInput
from service.backend.app.schemas import MetadataInput


def test_model_artifact_registry_loads_ready_manifest(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "time_series")

    record = ModelArtifactRegistry(tmp_path).get("time_series")

    assert record.ready
    assert record.manifest is not None
    assert record.manifest.model_name == "test_time_series_model"
    assert record.checkpoint_path is not None
    assert record.checkpoint_path.exists()


def test_checkpoint_timeseries_adapter_normalizes_backend_output(tmp_path: Path) -> None:
    record = _ready_record(tmp_path, "time_series")
    csv_path = tmp_path / "signal.csv"
    csv_path.write_text("0,1,2\n", encoding="utf-8")
    adapter = CheckpointTimeSeriesInferenceAdapter(record, backend=_FakeTimeSeriesBackend())

    result = adapter.run(TimeSeriesToolInput(csv_path=csv_path, csv_sha256="abc"))

    assert result.model_name == "test_time_series_model"
    assert result.label_id == 3
    assert result.label_name == "코로나 방전"
    assert result.features["pulse_rate"] == 0.12
    assert result.standard_evidence is not None


def test_checkpoint_vision_adapter_normalizes_backend_output(tmp_path: Path) -> None:
    record = _ready_record(tmp_path, "vision")
    image_path = tmp_path / "prpd.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter = CheckpointVisionInferenceAdapter(record, backend=_FakeVisionBackend())

    result = adapter.run(VisionToolInput(image_path=image_path, image_sha256="abc"))

    assert result.model_name == "test_vision_model"
    assert result.label_id == 1
    assert result.label_name == "노이즈"
    assert result.evidence["band_like_noise_score"] == 0.91
    assert result.standard_evidence is not None


def test_checkpoint_vlm_adapter_normalizes_backend_output(tmp_path: Path) -> None:
    record = _ready_record(tmp_path, "vlm")
    image_path = tmp_path / "prpd.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter = CheckpointVlmInferenceAdapter(record, backend=_FakeVlmBackend())

    result = adapter.run(
        VlmToolInput(
            image_path=image_path,
            image_sha256="abc",
            safe_metadata=_metadata(),
            timeseries_result=None,
            vision_result=None,
            rag_result=None,
        )
    )

    assert result.model_name == "test_vlm_model"
    assert result.label_id == 2
    assert result.diagnosis == "내부 방전"
    assert result.recommended_action == "현장 점검"
    assert result.standard_evidence is not None


def test_service_model_runtime_defaults_to_mock_adapters(tmp_path: Path) -> None:
    runtime = build_service_model_runtime(
        mock_adapters=MockModelAdapters(
            time_series=_MockAdapter("mock_ts"),
            vision=_MockAdapter("mock_vision"),
            vlm=_MockAdapter("mock_vlm"),
        ),
        settings=ModelAdapterSettings(mode="mock", artifact_root=tmp_path),
    )

    assert runtime.mode == "mock"
    assert runtime.time_series_info.adapter_kind == "mock"
    assert runtime.vision_info.adapter_kind == "mock"
    assert runtime.vlm_info.adapter_kind == "mock"


class _FakeTimeSeriesBackend:
    def predict_csv(self, tool_input: TimeSeriesToolInput) -> dict[str, object]:
        return {
            "label_id": 3,
            "confidence": 0.88,
            "probabilities": {"3": 0.88, "1": 0.12},
            "features": {"pulse_rate": 0.12, "abs_p99": 38.0, "spectral_energy": 1200.0},
        }


class _FakeVisionBackend:
    def predict_image(self, tool_input: VisionToolInput) -> dict[str, object]:
        return {
            "label_id": 1,
            "confidence": 0.81,
            "probabilities": {"1": 0.81, "3": 0.19},
            "evidence": {
                "band_like_noise_score": 0.91,
                "phase_localization_score": 0.09,
                "visual_evidence_summary": "노이즈 대역형 PRPD 패턴",
            },
        }


class _FakeVlmBackend:
    def generate_report(self, tool_input: VlmToolInput) -> dict[str, object]:
        return {
            "label_id": 2,
            "diagnosis": "내부 방전",
            "confidence": 0.79,
            "reason": "시계열/비전/RAG 근거를 종합한 내부 방전 판단",
            "recommended_action": "현장 점검",
        }


class _MockAdapter:
    def __init__(self, name: str) -> None:
        self.model_name = name
        self.model_version = "test"


def _ready_record(tmp_path: Path, task: ModelTask):
    _write_manifest(tmp_path, task)
    return ModelArtifactRegistry(tmp_path).get(task)


def _write_manifest(root: Path, task: ModelTask) -> None:
    task_dir = root / task
    task_dir.mkdir(parents=True)
    (task_dir / "checkpoint.pt").write_text("placeholder", encoding="utf-8")
    (task_dir / "preprocessor.json").write_text("{}", encoding="utf-8")
    manifest = {
        "task": task,
        "model_name": f"test_{task}_model",
        "model_version": "1.0.0",
        "framework": "test",
        "entrypoint": "tests.fake_model:load_adapter",
        "checkpoint_path": "checkpoint.pt",
        "preprocessor_path": "preprocessor.json",
        "input_spec": {"modality": task},
        "output_spec": {"required_fields": ["label_id", "confidence"]},
    }
    (task_dir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _metadata() -> MetadataInput:
    return MetadataInput(
        equipment_name="ACSR-OC",
        equipment_rated_voltage="22900V",
        equipment_rated_current="268A",
        sensor_type="HFCT",
        temperature=19,
        humidity=66,
    )
