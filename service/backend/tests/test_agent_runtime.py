from __future__ import annotations

from pathlib import Path

from service.backend.app.agent_runtime import AgentRunInput, LocalDiagnosisAgentRuntime
from service.backend.app.policy import label_name
from service.backend.app.schemas import TimeSeriesResult, VlmResult
from service.backend.app.tool_contracts import TimeSeriesToolInput, VlmToolInput


def test_local_agent_runtime_returns_needs_review_on_disagreement() -> None:
    runtime = LocalDiagnosisAgentRuntime(
        ts_adapter=_TsAdapter(label_id=2),
        vlm_adapter=_VlmAdapter(label_id=3),
    )

    result = runtime.run(
        AgentRunInput(
            diagnosis_id="diag_test",
            route="hybrid",
            timeseries_input=TimeSeriesToolInput(csv_path=Path("signal.csv"), csv_sha256="csvhash"),
            vlm_input=VlmToolInput(
                image_path=Path("image.png"),
                image_sha256="imagehash",
                safe_metadata=_metadata(),
                timeseries_result=None,
            ),
            rejection_reason=None,
        )
    )

    assert result.response.status == "needs_review"
    assert result.response.requires_human_review is True
    assert "time_series_tool" in [event["name"] for event in result.events]
    assert "vlm_tool" in [event["name"] for event in result.events]


class _TsAdapter:
    def __init__(self, label_id: int) -> None:
        self.label_id = label_id

    def run(self, tool_input: TimeSeriesToolInput) -> TimeSeriesResult:
        probabilities = {str(idx): 0.025 for idx in range(5)}
        probabilities[str(self.label_id)] = 0.90
        return TimeSeriesResult(
            model_name="test_ts",
            model_version="test",
            label_id=self.label_id,
            label_name=label_name(self.label_id),
            confidence=0.9,
            probabilities=probabilities,
            features={"rms": 1.0},
        )


class _VlmAdapter:
    def __init__(self, label_id: int) -> None:
        self.label_id = label_id

    def run(self, tool_input: VlmToolInput) -> VlmResult:
        return VlmResult(
            model_name="test_vlm",
            model_version="test",
            label_id=self.label_id,
            diagnosis=label_name(self.label_id),
            risk_level="주의",
            confidence=0.9,
            reason="test",
            recommended_action="test",
        )


def _metadata():
    from service.backend.app.schemas import MetadataInput

    return MetadataInput(
        equipment_name="ACSR-OC",
        equipment_rated_voltage="22900V",
        equipment_rated_current="268A",
        sensor_type="HFCT",
        temperature=19,
        humidity=66,
    )
