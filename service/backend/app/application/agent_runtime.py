from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from service.backend.app.application.contracts import (
    RagToolInput,
    SimilarCaseToolInput,
    TimeSeriesToolInput,
    VisionToolInput,
    VlmToolInput,
)
from service.backend.app.domain.fusion import build_fusion_summary
from service.backend.app.domain.guardrails import review_tool_outputs
from service.backend.app.domain.policy import recommended_action, risk_level
from service.backend.app.schemas import (
    DiagnosisResponse,
    DiagnosisRoute,
    FusionSummary,
    MetadataInput,
    RagResult,
    SimilarCaseResult,
    TimeSeriesResult,
    VisionResult,
    VlmResult,
)


@dataclass(frozen=True, slots=True)
class TraceEvent:
    name: str
    kind: str
    summary: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class TraceRecorder:
    events: list[TraceEvent] = field(default_factory=list)

    def record(self, name: str, kind: str, summary: dict[str, Any]) -> None:
        self.events.append(TraceEvent(name=name, kind=kind, summary=summary, created_at=_now_iso()))

    def public_events(self) -> list[dict[str, Any]]:
        return [
            {"name": event.name, "kind": event.kind, "summary": event.summary, "created_at": event.created_at}
            for event in self.events
        ]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class AgentRunInput:
    diagnosis_id: str
    route: DiagnosisRoute
    input_artifacts: dict[str, object]
    timeseries_input: TimeSeriesToolInput | None
    vision_input: VisionToolInput | None
    similar_case_input: SimilarCaseToolInput | None
    rag_input: RagToolInput | None
    vlm_input: VlmToolInput | None
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    response: DiagnosisResponse
    ts_result: TimeSeriesResult | None
    vision_result: VisionResult | None
    similar_case_result: SimilarCaseResult | None
    rag_result: RagResult | None
    vlm_result: VlmResult | None
    fusion_summary: FusionSummary | None
    events: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RuntimeAdapters:
    time_series: Any
    vision: Any
    similar_case: Any
    rag: Any
    vlm: Any


@dataclass(frozen=True, slots=True)
class ToolResults:
    time_series: TimeSeriesResult | None = None
    vision: VisionResult | None = None
    similar_case: SimilarCaseResult | None = None
    rag: RagResult | None = None
    vlm: VlmResult | None = None
    fusion: FusionSummary | None = None


@dataclass(frozen=True, slots=True)
class ReportInput:
    diagnosis_id: str
    trace_id: str
    route: DiagnosisRoute
    results: ToolResults
    trace: TraceRecorder


@dataclass(frozen=True, slots=True)
class RejectedResponseInput:
    diagnosis_id: str
    trace_id: str
    route: DiagnosisRoute
    reason: str


class LocalDiagnosisAgentRuntime:
    def __init__(self, adapters: RuntimeAdapters) -> None:
        self.adapters = adapters

    def run(self, run_input: AgentRunInput) -> AgentRunResult:
        diagnosis_id = run_input.diagnosis_id
        trace_id = f"trace_{uuid4().hex[:12]}"
        trace = TraceRecorder()
        trace.record("input_router", "guardrail", {"route": run_input.route})
        if run_input.input_artifacts:
            trace.record("input_artifacts", "context", run_input.input_artifacts)
        if run_input.route == "insufficient_input":
            response = _rejected_response(
                RejectedResponseInput(
                    diagnosis_id=diagnosis_id,
                    trace_id=trace_id,
                    route=run_input.route,
                    reason=run_input.rejection_reason or "invalid input",
                )
            )
            trace.record("input_rejected", "guardrail", {"reason": response.reason})
            return _agent_result(response, ToolResults(), trace)

        _record_metadata_context(run_input, trace)
        ts_result = self._run_timeseries_tool(run_input, trace)
        vision_result = self._run_vision_tool(run_input, trace)
        model_results = ToolResults(time_series=ts_result, vision=vision_result)
        similar_case_input = _with_case_evidence_results(run_input.similar_case_input, model_results)
        similar_case_result = self._run_similar_case_tool(run_input.route, similar_case_input, trace)
        retrieval_results = ToolResults(
            time_series=ts_result,
            vision=vision_result,
            similar_case=similar_case_result,
        )
        rag_input = _with_evidence_results(run_input.rag_input, retrieval_results)
        rag_result = self._run_rag_tool(run_input.route, rag_input, trace)
        report_results = ToolResults(
            time_series=ts_result,
            vision=vision_result,
            similar_case=similar_case_result,
            rag=rag_result,
        )
        vlm_input = _with_model_results(run_input.vlm_input, report_results)
        vlm_result = self._run_vlm_tool(run_input.route, vlm_input, trace)
        fusion_summary = build_fusion_summary(ts_result, vision_result, similar_case_result, rag_result, vlm_result)
        _record_fusion_summary(fusion_summary, trace)
        results = ToolResults(
            time_series=ts_result,
            vision=vision_result,
            similar_case=similar_case_result,
            rag=rag_result,
            vlm=vlm_result,
            fusion=fusion_summary,
        )
        response = _review_and_report(
            ReportInput(
                diagnosis_id=diagnosis_id,
                trace_id=trace_id,
                route=run_input.route,
                results=results,
                trace=trace,
            )
        )
        return _agent_result(response, results, trace)

    def _run_timeseries_tool(self, run_input: AgentRunInput, trace: TraceRecorder) -> TimeSeriesResult | None:
        if run_input.route not in {"timeseries_only", "hybrid"} or run_input.timeseries_input is None:
            return None
        result = self.adapters.time_series.run(run_input.timeseries_input)
        trace.record(
            "time_series_tool",
            "tool",
            {
                "model_name": result.model_name,
                "model_version": result.model_version,
                "label_id": result.label_id,
                "label_name": result.label_name,
                "confidence": round(result.confidence, 6),
                "csv_sha256": run_input.timeseries_input.csv_sha256,
                "features": result.features,
                "probabilities": result.probabilities,
                "standard_evidence": _dump_evidence(result.standard_evidence),
            },
        )
        return result

    def _run_vision_tool(self, run_input: AgentRunInput, trace: TraceRecorder) -> VisionResult | None:
        if run_input.route not in {"vlm_only", "hybrid"} or run_input.vision_input is None:
            return None
        result = self.adapters.vision.run(run_input.vision_input)
        trace.record(
            "vision_tool",
            "tool",
            {
                "model_name": result.model_name,
                "model_version": result.model_version,
                "label_id": result.label_id,
                "label_name": result.label_name,
                "confidence": round(result.confidence, 6),
                "image_sha256": run_input.vision_input.image_sha256,
                "evidence": result.evidence,
                "probabilities": result.probabilities,
                "standard_evidence": _dump_evidence(result.standard_evidence),
            },
        )
        return result

    def _run_similar_case_tool(
        self,
        route: DiagnosisRoute,
        tool_input: SimilarCaseToolInput | None,
        trace: TraceRecorder,
    ) -> SimilarCaseResult | None:
        if route == "insufficient_input" or tool_input is None:
            return None
        result = self.adapters.similar_case.run(tool_input)
        top_case = result.cases[0] if result.cases else None
        trace.record(
            "similar_case_tool",
            "tool",
            {
                "model_name": result.retriever_name,
                "model_version": result.retriever_version,
                "query": result.query,
                "case_count": len(result.cases),
                "top_sample_id": top_case.sample_id if top_case is not None else "n/a",
                "top_label": top_case.label_name if top_case is not None else "n/a",
                "top_similarity": top_case.similarity if top_case is not None else 0.0,
                "cases": [case.model_dump() for case in result.cases],
            },
        )
        return result

    def _run_rag_tool(self, route: DiagnosisRoute, tool_input: RagToolInput | None, trace: TraceRecorder) -> RagResult | None:
        if route == "insufficient_input" or tool_input is None:
            return None
        result = self.adapters.rag.run(tool_input)
        top_document = result.documents[0] if result.documents else None
        trace.record(
            "rag_tool",
            "tool",
            {
                "model_name": result.retriever_name,
                "model_version": result.retriever_version,
                "query": result.query,
                "document_count": len(result.documents),
                "top_document_id": top_document.document_id if top_document is not None else "n/a",
                "top_title": top_document.title if top_document is not None else "n/a",
                "documents": [document.model_dump() for document in result.documents],
                "similar_cases": [case.model_dump() for case in result.similar_cases],
            },
        )
        return result

    def _run_vlm_tool(self, route: DiagnosisRoute, tool_input: VlmToolInput | None, trace: TraceRecorder) -> VlmResult | None:
        if route not in {"vlm_only", "hybrid"} or tool_input is None:
            return None
        result = self.adapters.vlm.run(tool_input)
        trace.record(
            "vlm_tool",
            "tool",
            {
                "model_name": result.model_name,
                "model_version": result.model_version,
                "label_id": result.label_id,
                "label_name": result.diagnosis,
                "confidence": round(result.confidence, 6),
                "image_sha256": tool_input.image_sha256,
                "reason": result.reason,
                "recommended_action": result.recommended_action,
                "standard_evidence": _dump_evidence(result.standard_evidence),
            },
        )
        return result


def _with_model_results(
    tool_input: VlmToolInput | None,
    results: ToolResults,
) -> VlmToolInput | None:
    if tool_input is None:
        return None
    return VlmToolInput(
        image_path=tool_input.image_path,
        image_sha256=tool_input.image_sha256,
        safe_metadata=tool_input.safe_metadata,
        timeseries_result=results.time_series,
        vision_result=results.vision,
        rag_result=results.rag,
    )


def _with_evidence_results(
    tool_input: RagToolInput | None,
    results: ToolResults,
) -> RagToolInput | None:
    if tool_input is None:
        return None
    return RagToolInput(
        safe_metadata=tool_input.safe_metadata,
        route=tool_input.route,
        timeseries_result=results.time_series,
        vision_result=results.vision,
        similar_case_result=results.similar_case,
    )


def _with_case_evidence_results(
    tool_input: SimilarCaseToolInput | None,
    results: ToolResults,
) -> SimilarCaseToolInput | None:
    if tool_input is None:
        return None
    return SimilarCaseToolInput(
        safe_metadata=tool_input.safe_metadata,
        route=tool_input.route,
        timeseries_result=results.time_series,
        vision_result=results.vision,
        image_path=tool_input.image_path,
        timeseries_path=tool_input.timeseries_path,
    )


def _record_metadata_context(run_input: AgentRunInput, trace: TraceRecorder) -> None:
    metadata = _metadata_from_input(run_input)
    if metadata is None:
        return
    trace.record("metadata_context", "context", _metadata_summary(metadata))


def _metadata_from_input(run_input: AgentRunInput) -> MetadataInput | None:
    if run_input.rag_input is not None and run_input.rag_input.safe_metadata is not None:
        return run_input.rag_input.safe_metadata
    if run_input.vlm_input is not None:
        return run_input.vlm_input.safe_metadata
    return None


def _metadata_summary(metadata: MetadataInput) -> dict[str, Any]:
    return {
        "equipment_name": metadata.equipment_name,
        "equipment_type": metadata.equipment_type or "n/a",
        "rated_voltage": metadata.equipment_rated_voltage,
        "rated_current": metadata.equipment_rated_current,
        "sensor_type": metadata.sensor_type,
        "measurement_location": metadata.measurement_location or "n/a",
        "operating_condition": metadata.operating_condition or "n/a",
        "temperature": metadata.temperature,
        "humidity": metadata.humidity,
        "insulator_type": metadata.insulator_type or "n/a",
        "clearance_distance": metadata.clearance_distance or "n/a",
    }


def _record_fusion_summary(fusion_summary: FusionSummary, trace: TraceRecorder) -> None:
    trace.record(
        "fusion_engine",
        "fusion",
        {
            "strategy": fusion_summary.strategy,
            "final_label_id": fusion_summary.final_label_id,
            "final_label_name": fusion_summary.final_label_name,
            "confidence": fusion_summary.confidence,
            "agreement_level": fusion_summary.agreement_level,
            "contributing_sources": fusion_summary.contributing_sources,
            "rationale": fusion_summary.rationale,
            "evidence": [item.model_dump() for item in fusion_summary.evidence],
        },
    )


def _dump_evidence(evidence: object) -> dict[str, Any] | None:
    if evidence is None:
        return None
    if hasattr(evidence, "model_dump"):
        return evidence.model_dump()
    return None


def _review_and_report(report_input: ReportInput) -> DiagnosisResponse:
    results = report_input.results
    decision = review_tool_outputs(results.time_series, results.vision, results.vlm, results.rag)
    report_input.trace.record(
        "diagnosis_reviewer",
        "guardrail",
        {
            "status": decision.status,
            "requires_human_review": decision.requires_human_review,
            "reason": decision.reason,
            "fusion_agreement": results.fusion.agreement_level if results.fusion is not None else "none",
        },
    )
    if decision.status == "needs_review":
        return DiagnosisResponse(
            diagnosis_id=report_input.diagnosis_id,
            trace_id=report_input.trace_id,
            route=report_input.route,
            status="needs_review",
            reason=decision.reason,
            requires_human_review=True,
        )
    if decision.status == "rejected":
        return _rejected_response(
            RejectedResponseInput(
                diagnosis_id=report_input.diagnosis_id,
                trace_id=report_input.trace_id,
                route="insufficient_input",
                reason=decision.reason,
            )
        )

    final = results.vlm if results.vlm is not None else results.time_series or results.vision
    if final is None:
        return _rejected_response(
            RejectedResponseInput(
                diagnosis_id=report_input.diagnosis_id,
                trace_id=report_input.trace_id,
                route="insufficient_input",
                reason="insufficient input",
            )
        )
    label_id = final.label_id
    diagnosis = final.diagnosis if isinstance(final, VlmResult) else final.label_name
    report_input.trace.record(
        "report_agent",
        "agent",
        {
            "final_label_id": label_id,
            "diagnosis": diagnosis,
            "knowledge_documents": _rag_document_count(results.rag),
            "reference_cases": _rag_case_count(results.rag),
            "fusion_rationale": results.fusion.rationale if results.fusion is not None else "n/a",
        },
    )
    return DiagnosisResponse(
        diagnosis_id=report_input.diagnosis_id,
        trace_id=report_input.trace_id,
        route=report_input.route,
        status="completed",
        final_label_id=label_id,
        diagnosis=diagnosis,
        risk_level=risk_level(label_id),
        confidence=final.confidence,
        reason=decision.reason,
        recommended_action=recommended_action(label_id),
        requires_human_review=False,
    )


def _rag_document_count(rag_result: RagResult | None) -> int:
    return len(rag_result.documents) if rag_result is not None else 0


def _rag_case_count(rag_result: RagResult | None) -> int:
    return len(rag_result.similar_cases) if rag_result is not None else 0


def _agent_result(
    response: DiagnosisResponse,
    results: ToolResults,
    trace: TraceRecorder,
) -> AgentRunResult:
    return AgentRunResult(
        response=response,
        ts_result=results.time_series,
        vision_result=results.vision,
        similar_case_result=results.similar_case,
        rag_result=results.rag,
        vlm_result=results.vlm,
        fusion_summary=results.fusion,
        events=trace.public_events(),
    )


def _rejected_response(input_data: RejectedResponseInput) -> DiagnosisResponse:
    return DiagnosisResponse(
        diagnosis_id=input_data.diagnosis_id,
        trace_id=input_data.trace_id,
        route=input_data.route,
        status="rejected",
        reason=input_data.reason,
        requires_human_review=False,
        error_code="INVALID_INPUT",
    )
