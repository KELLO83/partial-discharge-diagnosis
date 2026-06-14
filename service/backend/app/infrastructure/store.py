from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from service.backend.app.schemas import (
    CaseTimelineEvent,
    DiagnosisCommentRecord,
    DiagnosisDetailResponse,
    DiagnosisListItem,
    DiagnosisRoute,
    DiagnosisStatus,
    ReviewActionRecord,
    TraceResponse,
)


@dataclass(frozen=True, slots=True)
class TraceRecord:
    diagnosis_id: str
    trace_id: str
    route: DiagnosisRoute
    status: DiagnosisStatus
    diagnosis: str | None
    risk_level: str | None
    confidence: float | None
    reason: str
    requires_human_review: bool
    created_at: str
    steps: tuple[str, ...]
    summary: dict[str, str]
    events: tuple[dict[str, object], ...] = ()


@dataclass(slots=True)
class TraceStore:
    records: dict[str, TraceRecord] = field(default_factory=dict)
    actions: dict[str, list[ReviewActionRecord]] = field(default_factory=dict)
    comments: dict[str, list[DiagnosisCommentRecord]] = field(default_factory=dict)

    def save(self, record: TraceRecord) -> None:
        self.records[record.diagnosis_id] = record

    def list(self, limit: int = 20) -> list[DiagnosisListItem]:
        return [_to_list_item(record) for record in self._recent_records(limit)]

    def review_queue(self, limit: int = 20) -> list[DiagnosisListItem]:
        queued = [
            record
            for record in self.records.values()
            if record.requires_human_review or record.status in {"needs_review", "rejected"}
        ]
        queued.sort(key=lambda record: record.created_at, reverse=True)
        return [_to_list_item(record) for record in queued[:limit]]

    def get(self, diagnosis_id: str) -> TraceResponse | None:
        record = self.records.get(diagnosis_id)
        if record is None:
            return None
        return TraceResponse(
            diagnosis_id=record.diagnosis_id,
            trace_id=record.trace_id,
            route=record.route,
            status=record.status,
            steps=list(record.steps),
            summary=record.summary,
            events=list(record.events),
        )

    def detail(self, diagnosis_id: str) -> DiagnosisDetailResponse | None:
        record = self.records.get(diagnosis_id)
        trace = self.get(diagnosis_id)
        if record is None or trace is None:
            return None
        return DiagnosisDetailResponse(
            diagnosis=_to_list_item(record),
            trace=trace,
            actions=list(self.actions.get(diagnosis_id, [])),
            comments=list(self.comments.get(diagnosis_id, [])),
            timeline=_timeline_for_record(
                record,
                actions=self.actions.get(diagnosis_id, []),
                comments=self.comments.get(diagnosis_id, []),
            ),
        )

    def add_action(self, diagnosis_id: str, action: str, note: str) -> ReviewActionRecord | None:
        if diagnosis_id not in self.records:
            return None
        record = ReviewActionRecord(action=action, note=note, created_at=now_iso())
        self.actions.setdefault(diagnosis_id, []).append(record)
        return record

    def add_comment(self, diagnosis_id: str, note: str) -> DiagnosisCommentRecord | None:
        if diagnosis_id not in self.records:
            return None
        record = DiagnosisCommentRecord(note=note, created_at=now_iso())
        self.comments.setdefault(diagnosis_id, []).append(record)
        return record

    def _recent_records(self, limit: int) -> list[TraceRecord]:
        records = list(self.records.values())
        records.sort(key=lambda record: record.created_at, reverse=True)
        return records[:limit]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _to_list_item(record: TraceRecord) -> DiagnosisListItem:
    return DiagnosisListItem(
        diagnosis_id=record.diagnosis_id,
        trace_id=record.trace_id,
        route=record.route,
        status=record.status,
        diagnosis=record.diagnosis,
        risk_level=record.risk_level,
        confidence=record.confidence,
        reason=record.reason,
        requires_human_review=record.requires_human_review,
        created_at=record.created_at,
    )


def _timeline_for_record(
    record: TraceRecord,
    actions: list[ReviewActionRecord],
    comments: list[DiagnosisCommentRecord],
) -> list[CaseTimelineEvent]:
    events = [
        CaseTimelineEvent(
            kind="diagnosis",
            title="진단 생성",
            body=f"{_route_label(record.route)} 경로가 {_status_label(record.status)} 상태를 반환했습니다.",
            created_at=record.created_at,
        )
    ]
    events.extend(_trace_timeline_events(record))
    events.extend(
        CaseTimelineEvent(
            kind="action",
            title=_action_label(action.action),
            body=action.note or "운영 조치가 기록되었습니다.",
            created_at=action.created_at,
        )
        for action in actions
    )
    events.extend(
        CaseTimelineEvent(
            kind="comment",
            title="운영 메모",
            body=comment.note,
            created_at=comment.created_at,
        )
        for comment in comments
    )
    return sorted(events, key=lambda event: event.created_at)


def _trace_timeline_events(record: TraceRecord) -> list[CaseTimelineEvent]:
    return [
        CaseTimelineEvent(
            kind="trace",
            title=_step_label(str(event.get("name", "trace event"))),
            body=_event_kind_label(str(event.get("kind", "agent"))),
            created_at=record.created_at,
        )
        for event in record.events
    ]


def _route_label(route: str) -> str:
    labels = {
        "hybrid": "종합 진단",
        "insufficient_input": "입력 대기",
        "timeseries_only": "시계열 진단",
        "vlm_only": "비전 진단",
    }
    return labels.get(route, route)


def _status_label(status: str) -> str:
    labels = {
        "completed": "완료",
        "needs_review": "검토",
        "rejected": "반려",
    }
    return labels.get(status, status)


def _action_label(action: str) -> str:
    labels = {
        "approve": "승인",
        "dispatch_field_team": "현장 출동",
        "mark_false_positive": "오탐 처리",
        "request_retest": "재측정 요청",
    }
    return labels.get(action, action)


def _step_label(step: str) -> str:
    labels = {
        "confidence_guardrail": "신뢰도 확인",
        "diagnosis_reviewer": "최종 검토",
        "disagreement_guardrail": "판정 불일치 확인",
        "fusion_engine": "근거 융합",
        "input_rejected": "입력 반려",
        "input_router": "입력 라우팅",
        "metadata_context": "설비 정보 정리",
        "rag_tool": "지식 검색",
        "report": "리포트 생성",
        "report_agent": "리포트 생성",
        "reviewer": "최종 검토",
        "similar_case_tool": "유사 사례 검색",
        "time_series_tool": "시계열 분석",
        "trace event": "처리 이벤트",
        "vision_tool": "비전 분석",
        "vlm_tool": "VLM 리포트",
    }
    return labels.get(step, step)


def _event_kind_label(kind: str) -> str:
    labels = {
        "agent": "에이전트",
        "context": "정보 정리",
        "fusion": "융합",
        "guardrail": "보호 규칙",
        "operation_scenario": "운영 시나리오",
        "router": "라우터",
        "tool": "모델/도구",
    }
    return labels.get(kind, kind)


trace_store = TraceStore()
