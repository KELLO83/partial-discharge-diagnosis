from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from service.backend.app.guardrails import review_tool_outputs
from service.backend.app.policy import recommended_action, risk_level
from service.backend.app.schemas import DiagnosisResponse, DiagnosisRoute, TimeSeriesResult, VlmResult
from service.backend.app.tool_contracts import TimeSeriesToolInput, VlmToolInput


@dataclass(frozen=True, slots=True)
class TraceEvent:
    name: str
    kind: str
    summary: dict[str, Any]


@dataclass(slots=True)
class TraceRecorder:
    events: list[TraceEvent] = field(default_factory=list)

    def record(self, name: str, kind: str, summary: dict[str, Any]) -> None:
        self.events.append(TraceEvent(name=name, kind=kind, summary=summary))

    def public_events(self) -> list[dict[str, Any]]:
        return [
            {"name": event.name, "kind": event.kind, "summary": event.summary}
            for event in self.events
        ]


@dataclass(frozen=True, slots=True)
class AgentRunInput:
    diagnosis_id: str
    route: DiagnosisRoute
    timeseries_input: TimeSeriesToolInput | None
    vlm_input: VlmToolInput | None
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    response: DiagnosisResponse
    ts_result: TimeSeriesResult | None
    vlm_result: VlmResult | None
    events: list[dict[str, Any]]


class LocalDiagnosisAgentRuntime:
    def __init__(self, ts_adapter: Any, vlm_adapter: Any) -> None:
        self.ts_adapter = ts_adapter
        self.vlm_adapter = vlm_adapter

    def run(self, run_input: AgentRunInput) -> AgentRunResult:
        diagnosis_id = run_input.diagnosis_id
        trace_id = f"trace_{uuid4().hex[:12]}"
        trace = TraceRecorder()
        trace.record("input_router", "guardrail", {"route": run_input.route})
        if run_input.route == "insufficient_input":
            response = _rejected_response(diagnosis_id, trace_id, run_input.route, run_input.rejection_reason or "invalid input")
            trace.record("input_rejected", "guardrail", {"reason": response.reason})
            return AgentRunResult(response, None, None, trace.public_events())

        ts_result = self._run_timeseries_tool(run_input, trace)
        vlm_input = _with_timeseries_result(run_input.vlm_input, ts_result)
        vlm_result = self._run_vlm_tool(run_input.route, vlm_input, trace)
        response = _review_and_report(diagnosis_id, trace_id, run_input.route, ts_result, vlm_result, trace)
        return AgentRunResult(response, ts_result, vlm_result, trace.public_events())

    def _run_timeseries_tool(self, run_input: AgentRunInput, trace: TraceRecorder) -> TimeSeriesResult | None:
        if run_input.route not in {"timeseries_only", "hybrid"} or run_input.timeseries_input is None:
            return None
        result = self.ts_adapter.run(run_input.timeseries_input)
        trace.record(
            "time_series_tool",
            "tool",
            {
                "model_name": result.model_name,
                "model_version": result.model_version,
                "label_id": result.label_id,
                "confidence": round(result.confidence, 6),
                "csv_sha256": run_input.timeseries_input.csv_sha256,
            },
        )
        return result

    def _run_vlm_tool(self, route: DiagnosisRoute, tool_input: VlmToolInput | None, trace: TraceRecorder) -> VlmResult | None:
        if route not in {"vlm_only", "hybrid"} or tool_input is None:
            return None
        result = self.vlm_adapter.run(tool_input)
        trace.record(
            "vlm_tool",
            "tool",
            {
                "model_name": result.model_name,
                "model_version": result.model_version,
                "label_id": result.label_id,
                "confidence": round(result.confidence, 6),
                "image_sha256": tool_input.image_sha256,
            },
        )
        return result


def _with_timeseries_result(tool_input: VlmToolInput | None, ts_result: TimeSeriesResult | None) -> VlmToolInput | None:
    if tool_input is None:
        return None
    return VlmToolInput(
        image_path=tool_input.image_path,
        image_sha256=tool_input.image_sha256,
        safe_metadata=tool_input.safe_metadata,
        timeseries_result=ts_result,
    )


def _review_and_report(
    diagnosis_id: str,
    trace_id: str,
    route: DiagnosisRoute,
    ts_result: TimeSeriesResult | None,
    vlm_result: VlmResult | None,
    trace: TraceRecorder,
) -> DiagnosisResponse:
    decision = review_tool_outputs(ts_result, vlm_result)
    trace.record(
        "diagnosis_reviewer",
        "guardrail",
        {
            "status": decision.status,
            "requires_human_review": decision.requires_human_review,
            "reason": decision.reason,
        },
    )
    if decision.status == "needs_review":
        return DiagnosisResponse(
            diagnosis_id=diagnosis_id,
            trace_id=trace_id,
            route=route,
            status="needs_review",
            reason=decision.reason,
            requires_human_review=True,
        )
    if decision.status == "rejected":
        return _rejected_response(diagnosis_id, trace_id, "insufficient_input", decision.reason)

    final = vlm_result if vlm_result is not None else ts_result
    if final is None:
        return _rejected_response(diagnosis_id, trace_id, "insufficient_input", "insufficient input")
    label_id = final.label_id
    diagnosis = final.diagnosis if isinstance(final, VlmResult) else final.label_name
    trace.record("report_agent", "agent", {"final_label_id": label_id, "diagnosis": diagnosis})
    return DiagnosisResponse(
        diagnosis_id=diagnosis_id,
        trace_id=trace_id,
        route=route,
        status="completed",
        final_label_id=label_id,
        diagnosis=diagnosis,
        risk_level=risk_level(label_id),
        confidence=final.confidence,
        reason=decision.reason,
        recommended_action=recommended_action(label_id),
        requires_human_review=False,
    )


def _rejected_response(diagnosis_id: str, trace_id: str, route: DiagnosisRoute, reason: str) -> DiagnosisResponse:
    return DiagnosisResponse(
        diagnosis_id=diagnosis_id,
        trace_id=trace_id,
        route=route,
        status="rejected",
        reason=reason,
        requires_human_review=False,
        error_code="INVALID_INPUT",
    )
