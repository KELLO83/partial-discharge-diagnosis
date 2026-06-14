from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from service.backend.app.application.agent_runtime import AgentRunInput, LocalDiagnosisAgentRuntime, RuntimeAdapters
from service.backend.app.schemas import DiagnosisResponse, DiagnosisRoute, FusionSummary, MetadataInput, RagResult, SimilarCaseResult, TimeSeriesResult, VisionResult, VlmResult
from service.backend.app.infrastructure.store import TraceRecord, now_iso, trace_store
from service.backend.app.application.contracts import RagToolInput, SimilarCaseToolInput, TimeSeriesToolInput, VisionToolInput, VlmToolInput
from service.backend.app.application.adapters import rag_adapter, similar_case_adapter, time_series_adapter, vision_adapter, vlm_adapter
from service.backend.app.infrastructure.validation import CsvShape


@dataclass(frozen=True, slots=True)
class WorkflowInput:
    diagnosis_id: str
    has_png_image: bool
    csv_shape: CsvShape | None
    metadata: MetadataInput | None
    metadata_error: str | None
    image_path: Path | None = None
    image_sha256: str | None = None
    csv_path: Path | None = None
    csv_sha256: str | None = None
    input_artifacts: dict[str, object] = field(default_factory=dict)


def run_diagnosis_workflow(input_data: WorkflowInput) -> DiagnosisResponse:
    route = _select_route(input_data)
    runtime = LocalDiagnosisAgentRuntime(
        RuntimeAdapters(
            time_series=time_series_adapter,
            vision=vision_adapter,
            similar_case=similar_case_adapter,
            rag=rag_adapter,
            vlm=vlm_adapter,
        )
    )
    result = runtime.run(
        AgentRunInput(
            route=route,
            diagnosis_id=input_data.diagnosis_id,
            input_artifacts=input_data.input_artifacts,
            timeseries_input=_timeseries_tool_input(input_data),
            vision_input=_vision_tool_input(input_data),
            similar_case_input=_similar_case_tool_input(input_data, route),
            rag_input=_rag_tool_input(input_data, route),
            vlm_input=_vlm_tool_input(input_data),
            rejection_reason=_rejection_reason(input_data),
        )
    )
    _save_trace(
        result.response,
        result.events,
        input_data.metadata,
        result.ts_result,
        result.vision_result,
        result.similar_case_result,
        result.rag_result,
        result.vlm_result,
        result.fusion_summary,
    )
    return result.response


def run_diagnosis_workflow_with_runtime(
    input_data: WorkflowInput,
    runtime: LocalDiagnosisAgentRuntime,
) -> DiagnosisResponse:
    route = _select_route(input_data)
    result = runtime.run(
        AgentRunInput(
            route=route,
            diagnosis_id=input_data.diagnosis_id,
            input_artifacts=input_data.input_artifacts,
            timeseries_input=_timeseries_tool_input(input_data),
            vision_input=_vision_tool_input(input_data),
            similar_case_input=_similar_case_tool_input(input_data, route),
            rag_input=_rag_tool_input(input_data, route),
            vlm_input=_vlm_tool_input(input_data),
            rejection_reason=_rejection_reason(input_data),
        )
    )
    _save_trace(
        result.response,
        result.events,
        input_data.metadata,
        result.ts_result,
        result.vision_result,
        result.similar_case_result,
        result.rag_result,
        result.vlm_result,
        result.fusion_summary,
    )
    return result.response


def parse_metadata(raw: str | None) -> tuple[MetadataInput | None, str | None]:
    if raw is None or raw.strip() == "":
        return None, None
    try:
        return MetadataInput.model_validate_json(raw), None
    except ValidationError:
        return None, "metadata must contain required equipment and environment fields"


def _select_route(input_data: WorkflowInput) -> Literal["insufficient_input", "timeseries_only", "vlm_only", "hybrid"]:
    has_valid_csv = input_data.csv_shape is not None and input_data.csv_shape.valid
    has_metadata = input_data.metadata is not None
    if input_data.has_png_image and has_metadata and has_valid_csv:
        return "hybrid"
    if input_data.has_png_image and has_metadata:
        return "vlm_only"
    if has_valid_csv:
        return "timeseries_only"
    return "insufficient_input"


def _timeseries_tool_input(input_data: WorkflowInput) -> TimeSeriesToolInput | None:
    if input_data.csv_path is None or input_data.csv_sha256 is None:
        return None
    return TimeSeriesToolInput(csv_path=input_data.csv_path, csv_sha256=input_data.csv_sha256)


def _vision_tool_input(input_data: WorkflowInput) -> VisionToolInput | None:
    if input_data.image_path is None or input_data.image_sha256 is None:
        return None
    return VisionToolInput(image_path=input_data.image_path, image_sha256=input_data.image_sha256)


def _vlm_tool_input(input_data: WorkflowInput) -> VlmToolInput | None:
    if input_data.image_path is None or input_data.image_sha256 is None or input_data.metadata is None:
        return None
    return VlmToolInput(
        image_path=input_data.image_path,
        image_sha256=input_data.image_sha256,
        safe_metadata=input_data.metadata,
        timeseries_result=None,
        vision_result=None,
        rag_result=None,
    )


def _rag_tool_input(input_data: WorkflowInput, route: DiagnosisRoute) -> RagToolInput | None:
    if route == "insufficient_input":
        return None
    return RagToolInput(
        safe_metadata=input_data.metadata,
        route=route,
        timeseries_result=None,
        vision_result=None,
        similar_case_result=None,
    )


def _similar_case_tool_input(input_data: WorkflowInput, route: DiagnosisRoute) -> SimilarCaseToolInput | None:
    if route == "insufficient_input":
        return None
    return SimilarCaseToolInput(
        safe_metadata=input_data.metadata,
        route=route,
        timeseries_result=None,
        vision_result=None,
        image_path=input_data.image_path,
        timeseries_path=input_data.csv_path,
    )


def _save_trace(
    response: DiagnosisResponse,
    events: list[dict[str, object]],
    metadata: MetadataInput | None,
    ts_result: TimeSeriesResult | None,
    vision_result: VisionResult | None,
    similar_case_result: SimilarCaseResult | None,
    rag_result: RagResult | None,
    vlm_result: VlmResult | None,
    fusion_summary: FusionSummary | None,
) -> None:
    summary: dict[str, str] = {"reason": response.reason}
    if metadata is not None:
        summary["metadata"] = _metadata_summary(metadata)
    if ts_result is not None:
        summary["time_series"] = f"{ts_result.label_name}:{ts_result.confidence:.2f}"
    if vision_result is not None:
        summary["vision"] = f"{vision_result.label_name}:{vision_result.confidence:.2f}"
    if similar_case_result is not None:
        summary["similar_cases"] = f"{len(similar_case_result.cases)} cases:{similar_case_result.retriever_name}"
    if rag_result is not None:
        summary["rag"] = f"{len(rag_result.documents)} docs/{len(rag_result.similar_cases)} cases:{rag_result.retriever_name}"
    if vlm_result is not None:
        summary["vlm"] = f"{vlm_result.diagnosis}:{vlm_result.confidence:.2f}"
    if fusion_summary is not None:
        summary["fusion"] = f"{fusion_summary.agreement_level}:{fusion_summary.final_label_name or 'n/a'}"
    trace_store.save(
        TraceRecord(
            diagnosis_id=response.diagnosis_id,
            trace_id=response.trace_id,
            route=response.route,
            status=response.status,
            diagnosis=response.diagnosis,
            risk_level=response.risk_level,
            confidence=response.confidence,
            reason=response.reason,
            requires_human_review=response.requires_human_review,
            created_at=now_iso(),
            steps=tuple(str(event.get("name", "")) for event in events),
            summary=summary,
            events=tuple(events),
        )
    )


def _rejection_reason(input_data: WorkflowInput) -> str:
    if input_data.csv_shape is not None and not input_data.csv_shape.valid:
        return input_data.csv_shape.message
    if input_data.metadata_error is not None:
        return input_data.metadata_error
    return "PRPD 이미지+metadata 또는 시계열 CSV 중 하나 이상이 필요합니다."


def _metadata_summary(metadata: MetadataInput) -> str:
    parts = [
        metadata.equipment_name,
        metadata.equipment_type or "unknown type",
        metadata.equipment_rated_voltage,
        metadata.sensor_type,
        metadata.measurement_location or "unknown location",
    ]
    return " / ".join(parts)
