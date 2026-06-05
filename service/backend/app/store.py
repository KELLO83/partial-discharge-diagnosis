from __future__ import annotations

from dataclasses import dataclass, field

from service.backend.app.schemas import DiagnosisRoute, DiagnosisStatus, TraceResponse


@dataclass(frozen=True, slots=True)
class TraceRecord:
    diagnosis_id: str
    trace_id: str
    route: DiagnosisRoute
    status: DiagnosisStatus
    steps: tuple[str, ...]
    summary: dict[str, str]


@dataclass(slots=True)
class TraceStore:
    records: dict[str, TraceRecord] = field(default_factory=dict)

    def save(self, record: TraceRecord) -> None:
        self.records[record.diagnosis_id] = record

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
        )


trace_store = TraceStore()
