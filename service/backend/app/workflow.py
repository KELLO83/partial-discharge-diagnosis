from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from pydantic import ValidationError

from service.backend.app.schemas import DiagnosisResponse, DiagnosisRoute, MetadataInput, TimeSeriesResult, VlmResult
from service.backend.app.store import TraceRecord, trace_store
from service.backend.app.tools import run_timeseries_inference, run_vlm_inference
from service.backend.app.validation import CsvShape


@dataclass(frozen=True, slots=True)
class WorkflowInput:
    has_png_image: bool
    csv_shape: CsvShape | None
    metadata: MetadataInput | None
    metadata_error: str | None


def run_diagnosis_workflow(input_data: WorkflowInput) -> DiagnosisResponse:
    diagnosis_id = f"diag_{uuid4().hex[:12]}"
    trace_id = f"trace_{uuid4().hex[:12]}"
    steps = ["input_router"]
    route = _select_route(input_data)
    if route == "insufficient_input":
        return _reject(diagnosis_id, trace_id, route, steps, _rejection_reason(input_data))
    ts_result = _run_ts_if_needed(route, steps)
    vlm_result = _run_vlm_if_needed(route, steps, input_data.metadata, ts_result)
    response = _review(diagnosis_id, trace_id, route, steps, ts_result, vlm_result)
    _save_trace(response, steps, ts_result, vlm_result)
    return response


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


def _run_ts_if_needed(route: DiagnosisRoute, steps: list[str]) -> TimeSeriesResult | None:
    if route not in {"timeseries_only", "hybrid"}:
        return None
    steps.append("time_series_tool")
    return run_timeseries_inference()


def _run_vlm_if_needed(
    route: DiagnosisRoute,
    steps: list[str],
    metadata: MetadataInput | None,
    ts_result: TimeSeriesResult | None,
) -> VlmResult | None:
    if route not in {"vlm_only", "hybrid"} or metadata is None:
        return None
    steps.append("vlm_tool")
    return run_vlm_inference(metadata, ts_result)


def _review(
    diagnosis_id: str,
    trace_id: str,
    route: DiagnosisRoute,
    steps: list[str],
    ts_result: TimeSeriesResult | None,
    vlm_result: VlmResult | None,
) -> DiagnosisResponse:
    steps.append("diagnosis_reviewer")
    if ts_result is not None and vlm_result is not None and ts_result.label_id != vlm_result.label_id:
        return DiagnosisResponse(
            diagnosis_id=diagnosis_id,
            trace_id=trace_id,
            route="hybrid",
            status="needs_review",
            reason="시계열 모델과 VLM의 예측 라벨이 불일치합니다.",
            requires_human_review=True,
        )
    final = vlm_result if vlm_result is not None else ts_result
    if final is None:
        return _reject(diagnosis_id, trace_id, "insufficient_input", steps, "insufficient input")
    label_id = final.label_id
    diagnosis = final.diagnosis if isinstance(final, VlmResult) else final.label_name
    confidence = final.confidence
    return DiagnosisResponse(
        diagnosis_id=diagnosis_id,
        trace_id=trace_id,
        route=route,
        status="completed",
        final_label_id=label_id,
        diagnosis=diagnosis,
        risk_level="주의" if label_id >= 2 else "낮음",
        confidence=confidence,
        reason="tool 기반 추론 결과가 일관되어 최종 진단을 확정했습니다.",
        recommended_action=_action_for_label(label_id),
        requires_human_review=False,
    )


def _reject(
    diagnosis_id: str,
    trace_id: str,
    route: DiagnosisRoute,
    steps: list[str],
    reason: str,
) -> DiagnosisResponse:
    response = DiagnosisResponse(
        diagnosis_id=diagnosis_id,
        trace_id=trace_id,
        route=route,
        status="rejected",
        reason=reason,
        requires_human_review=False,
        error_code="INVALID_INPUT",
    )
    _save_trace(response, steps, None, None)
    return response


def _save_trace(
    response: DiagnosisResponse,
    steps: list[str],
    ts_result: TimeSeriesResult | None,
    vlm_result: VlmResult | None,
) -> None:
    summary: dict[str, str] = {"reason": response.reason}
    if ts_result is not None:
        summary["time_series"] = f"{ts_result.label_name}:{ts_result.confidence:.2f}"
    if vlm_result is not None:
        summary["vlm"] = f"{vlm_result.diagnosis}:{vlm_result.confidence:.2f}"
    trace_store.save(
        TraceRecord(
            diagnosis_id=response.diagnosis_id,
            trace_id=response.trace_id,
            route=response.route,
            status=response.status,
            steps=tuple(steps),
            summary=summary,
        )
    )


def _rejection_reason(input_data: WorkflowInput) -> str:
    if input_data.csv_shape is not None and not input_data.csv_shape.valid:
        return input_data.csv_shape.message
    if input_data.metadata_error is not None:
        return input_data.metadata_error
    return "PRPD 이미지+metadata 또는 시계열 CSV 중 하나 이상이 필요합니다."


def _action_for_label(label_id: int) -> str:
    match label_id:
        case 0:
            return "정상 상태로 판단되며 정기 모니터링을 유지하세요."
        case 1:
            return "센서 접촉 상태와 주변 전자기 간섭 가능성을 점검하세요."
        case 2:
            return "절연체 표면 오염과 트래킹 흔적을 점검하세요."
        case 3:
            return "고전압 접속부와 전계 집중 부위를 점검하세요."
        case 4:
            return "절연체 내부 결함 가능성을 고려해 정밀 진단을 진행하세요."
        case _:
            return "추가 데이터 확인 후 재진단하세요."
