from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from service.backend.app.schemas import DiagnosisListItem, DiagnosisRoute, DiagnosisStatus, TraceResponse


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


trace_store = TraceStore()
