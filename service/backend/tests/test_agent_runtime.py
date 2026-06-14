from __future__ import annotations

from pathlib import Path

from service.backend.app.application.agent_runtime import AgentRunInput, LocalDiagnosisAgentRuntime, RuntimeAdapters
from service.backend.app.domain.policy import label_name
from service.backend.app.application.contracts import RagToolInput, SimilarCaseToolInput, TimeSeriesToolInput, VisionToolInput, VlmToolInput
from service.backend.app.schemas import (
    MetadataInput,
    RagDocument,
    RagResult,
    SimilarCase,
    SimilarCaseResult,
    TimeSeriesResult,
    VisionResult,
    VlmResult,
)


def test_local_agent_runtime_returns_needs_review_on_disagreement() -> None:
    runtime = LocalDiagnosisAgentRuntime(
        RuntimeAdapters(
            time_series=_TsAdapter(label_id=2),
            vision=_VisionAdapter(label_id=2),
            similar_case=_SimilarCaseAdapter(),
            rag=_RagAdapter(),
            vlm=_VlmAdapter(label_id=3),
        )
    )

    result = runtime.run(
        AgentRunInput(
            diagnosis_id="diag_test",
            route="hybrid",
            input_artifacts={},
            timeseries_input=TimeSeriesToolInput(csv_path=Path("signal.csv"), csv_sha256="csvhash"),
            vision_input=VisionToolInput(image_path=Path("image.png"), image_sha256="imagehash"),
            similar_case_input=SimilarCaseToolInput(
                safe_metadata=_metadata(),
                route="hybrid",
                timeseries_result=None,
                vision_result=None,
            ),
            rag_input=RagToolInput(
                safe_metadata=_metadata(),
                route="hybrid",
                timeseries_result=None,
                vision_result=None,
                similar_case_result=None,
            ),
            vlm_input=VlmToolInput(
                image_path=Path("image.png"),
                image_sha256="imagehash",
                safe_metadata=_metadata(),
                timeseries_result=None,
                vision_result=None,
                rag_result=None,
            ),
            rejection_reason=None,
        )
    )

    assert result.response.status == "needs_review"
    assert result.response.requires_human_review is True
    assert "time_series_tool" in [event["name"] for event in result.events]
    assert "vision_tool" in [event["name"] for event in result.events]
    assert "similar_case_tool" in [event["name"] for event in result.events]
    assert "rag_tool" in [event["name"] for event in result.events]
    assert "vlm_tool" in [event["name"] for event in result.events]
    assert "fusion_engine" in [event["name"] for event in result.events]


class _TsAdapter:
    def __init__(self, label_id: int) -> None:
        self.label_id = label_id

    def run(self, tool_input: TimeSeriesToolInput) -> TimeSeriesResult:
        return TimeSeriesResult(
            model_name="test_ts",
            model_version="test",
            label_id=self.label_id,
            label_name=label_name(self.label_id),
            confidence=0.9,
            probabilities=_probabilities(self.label_id),
            features={"rms": 1.0},
        )


class _VisionAdapter:
    def __init__(self, label_id: int) -> None:
        self.label_id = label_id

    def run(self, tool_input: VisionToolInput) -> VisionResult:
        return VisionResult(
            model_name="test_vision",
            model_version="test",
            label_id=self.label_id,
            label_name=label_name(self.label_id),
            confidence=0.9,
            probabilities=_probabilities(self.label_id),
            evidence={"phase_uniformity_score": 0.2},
        )


class _RagAdapter:
    def run(self, tool_input: RagToolInput) -> RagResult:
        return RagResult(
            retriever_name="test_rag",
            retriever_version="test",
            query=f"route={tool_input.route}",
            documents=[
                RagDocument(
                    document_id="test-rule",
                    title="test rule",
                    source="test",
                    excerpt="test excerpt",
                    relevance=0.9,
                )
            ],
            similar_cases=tool_input.similar_case_result.cases if tool_input.similar_case_result is not None else [],
        )


class _SimilarCaseAdapter:
    def run(self, tool_input: SimilarCaseToolInput) -> SimilarCaseResult:
        return SimilarCaseResult(
            retriever_name="test_case_retriever",
            retriever_version="test",
            query=f"route={tool_input.route}",
            cases=[
                SimilarCase(
                    sample_id="case-1",
                    label_id=2,
                    label_name=label_name(2),
                    equipment_name="ACSR-OC",
                    insulator_type="고체",
                    sensor_type="HFCT",
                    clearance_distance="1000mm",
                    similarity=0.9,
                    reason="test case",
                    image_url="/dataset/cases/case-1/image",
                    metadata={"max_discharge_value": 82.0},
                )
            ],
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


def _metadata() -> MetadataInput:
    return MetadataInput(
        equipment_name="ACSR-OC",
        equipment_rated_voltage="22900V",
        equipment_rated_current="268A",
        sensor_type="HFCT",
        temperature=19,
        humidity=66,
    )


def _probabilities(label_id: int) -> dict[str, float]:
    probabilities = {str(idx): 0.025 for idx in range(5)}
    probabilities[str(label_id)] = 0.90
    return probabilities
