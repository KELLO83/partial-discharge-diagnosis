from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from service.backend.app.schemas import (
    DemoScenario,
    DemoSeedResponse,
    DiagnosisRoute,
    DiagnosisStatus,
    ReviewActionRecord,
)
from service.backend.app.infrastructure.store import TraceRecord, now_iso, trace_store


@dataclass(frozen=True, slots=True)
class DemoCase:
    scenario_id: str
    title: str
    diagnosis_id: str
    trace_id: str
    route: str
    status: str
    diagnosis: str | None
    risk_level: str | None
    confidence: float | None
    reason: str
    requires_human_review: bool
    summary: str
    steps: tuple[str, ...]
    actions: tuple[tuple[str, str], ...] = ()


DEMO_CASES: tuple[DemoCase, ...] = (
    DemoCase(
        scenario_id="normal_baseline",
        title="정상 기준 상태",
        diagnosis_id="demo_normal_0001",
        trace_id="trace_demo_normal_0001",
        route="hybrid",
        status="completed",
        diagnosis="정상",
        risk_level="낮음",
        confidence=0.94,
        reason="시계열, 경량 비전, 유사 사례, 지식 검색, PRPD 리포트 근거가 모두 정상 운전 상태와 일치합니다.",
        requires_human_review=False,
        summary="관리자가 정상 운전 상태와 처리 추적을 확인하는 기준 사례입니다.",
        steps=("input_router", "metadata_context", "time_series_tool", "vision_tool", "similar_case_tool", "rag_tool", "vlm_tool", "reviewer", "report"),
    ),
    DemoCase(
        scenario_id="corona_caution",
        title="코로나 방전 주의",
        diagnosis_id="demo_corona_0001",
        trace_id="trace_demo_corona_0001",
        route="hybrid",
        status="completed",
        diagnosis="코로나 방전",
        risk_level="주의",
        confidence=0.88,
        reason="시계열, 경량 비전, 유사 사례, 지식 검색 근거가 고전압 접속부 인근의 코로나 방전 가능성을 지지합니다.",
        requires_human_review=False,
        summary="참조 근거와 정비 조치가 함께 표시되는 주의 사례입니다.",
        steps=("input_router", "metadata_context", "time_series_tool", "vision_tool", "similar_case_tool", "rag_tool", "vlm_tool", "reviewer", "report"),
        actions=(("approve", "엔지니어 검토 후 운영자가 승인했습니다."),),
    ),
    DemoCase(
        scenario_id="void_high_risk",
        title="보이드 방전 검토",
        diagnosis_id="demo_void_0001",
        trace_id="trace_demo_void_0001",
        route="timeseries_only",
        status="needs_review",
        diagnosis="보이드 방전",
        risk_level="높음",
        confidence=0.63,
        reason="시계열 모델은 보이드 방전을 지시하지만 신뢰도가 자동 확정 기준보다 낮습니다.",
        requires_human_review=True,
        summary="검토 대기열과 현장 출동 판단을 확인하는 고위험 사례입니다.",
        steps=("input_router", "time_series_tool", "similar_case_tool", "rag_tool", "confidence_guardrail", "reviewer"),
        actions=(("dispatch_field_team", "정비 계획 수립을 위해 현장 점검으로 인계했습니다."),),
    ),
    DemoCase(
        scenario_id="model_disagreement",
        title="모델 판단 불일치",
        diagnosis_id="demo_disagree_0001",
        trace_id="trace_demo_disagree_0001",
        route="hybrid",
        status="needs_review",
        diagnosis="표면 방전",
        risk_level="주의",
        confidence=0.71,
        reason="시계열, 비전, 유사 사례, 지식 검색, VLM 경로의 판단이 일치하지 않아 최종 판정을 보류했습니다.",
        requires_human_review=True,
        summary="근거가 서로 다를 때 운영자 검토로 전환되는 흐름을 보여줍니다.",
        steps=("input_router", "metadata_context", "time_series_tool", "vision_tool", "similar_case_tool", "rag_tool", "vlm_tool", "disagreement_guardrail", "reviewer"),
        actions=(("request_retest", "운영자가 PRPD 재측정을 요청했습니다."),),
    ),
    DemoCase(
        scenario_id="invalid_input",
        title="입력 반려",
        diagnosis_id="demo_invalid_0001",
        trace_id="trace_demo_invalid_0001",
        route="insufficient_input",
        status="rejected",
        diagnosis=None,
        risk_level=None,
        confidence=None,
        reason="모델 실행 전에 CSV 형식 확인에서 반려되었습니다.",
        requires_human_review=False,
        summary="입력이 부족하거나 잘못된 경우 모델을 실행하지 않는 보호 흐름입니다.",
        steps=("input_router", "input_rejected"),
    ),
)


def demo_scenarios() -> list[DemoScenario]:
    return [_to_scenario(case) for case in DEMO_CASES]


def seed_demo_records() -> DemoSeedResponse:
    seeded = []
    for case in DEMO_CASES:
        _seed_case(case)
        seeded.append(case.diagnosis_id)
    return DemoSeedResponse(seeded=seeded, scenarios=demo_scenarios())


def activate_demo_scenario(scenario_id: str) -> tuple[DemoScenario, str] | None:
    case = _find_case(scenario_id)
    if case is None:
        return None
    _seed_case(case)
    return _to_scenario(case), case.diagnosis_id


def _find_case(scenario_id: str) -> DemoCase | None:
    return next((case for case in DEMO_CASES if case.scenario_id == scenario_id), None)


def _seed_case(case: DemoCase) -> None:
    created_at = now_iso()
    trace_store.save(
        TraceRecord(
            diagnosis_id=case.diagnosis_id,
            trace_id=case.trace_id,
            route=cast(DiagnosisRoute, case.route),
            status=cast(DiagnosisStatus, case.status),
            diagnosis=case.diagnosis,
            risk_level=case.risk_level,
            confidence=case.confidence,
            reason=case.reason,
            requires_human_review=case.requires_human_review,
            created_at=created_at,
            steps=case.steps,
            summary={"reason": case.reason, "source": "operation_scenario"},
            events=tuple(_demo_event(step, case) for step in case.steps),
        )
    )
    trace_store.actions[case.diagnosis_id] = [
        ReviewActionRecord(action=action, note=note, created_at=created_at)
        for action, note in case.actions
    ]
    trace_store.comments.setdefault(case.diagnosis_id, [])


def _demo_event(step: str, case: DemoCase) -> dict[str, object]:
    return {
        "name": step,
        "kind": "operation_scenario",
        "summary": {
            "scenario_id": case.scenario_id,
            "diagnosis": case.diagnosis or "없음",
            "risk_level": case.risk_level or "없음",
            "source": "operation_scenario",
        },
    }


def _to_scenario(case: DemoCase) -> DemoScenario:
    return DemoScenario(
        scenario_id=case.scenario_id,
        title=case.title,
        diagnosis_id=case.diagnosis_id,
        route=cast(DiagnosisRoute, case.route),
        status=cast(DiagnosisStatus, case.status),
        risk_level=case.risk_level,
        summary=case.summary,
    )
