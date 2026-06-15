from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from service.backend.app.config.env import load_project_env
from service.backend.app.rag.settings import DEFAULT_DATABASE_URL
from service.backend.app.schemas import (
    CaseTimelineEvent,
    DiagnosisDetailResponse,
    DiagnosisListItem,
    DiagnosisRoute,
    DiagnosisStatus,
    TraceResponse,
)

DIAGNOSIS_SCHEMA = "diagnosis"


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
    database_url: str | None = field(default_factory=lambda: _database_url_from_env())
    database_path: Path | None = None

    def __post_init__(self) -> None:
        if self.database_path is not None:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        self._load_database()

    def save(self, record: TraceRecord) -> None:
        self.records[record.diagnosis_id] = record
        self._save_record(record)

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
        self._ensure_record_loaded(diagnosis_id)
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
        self._ensure_record_loaded(diagnosis_id)
        record = self.records.get(diagnosis_id)
        trace = self.get(diagnosis_id)
        if record is None or trace is None:
            return None
        return DiagnosisDetailResponse(
            diagnosis=_to_list_item(record),
            trace=trace,
            timeline=_timeline_for_record(record),
        )

    def _recent_records(self, limit: int) -> list[TraceRecord]:
        records = list(self.records.values())
        records.sort(key=lambda record: record.created_at, reverse=True)
        return records[:limit]

    def _connect_sqlite(self) -> sqlite3.Connection:
        if self.database_path is None:
            raise RuntimeError("SQLite database path is not configured")
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _connect_postgres(self) -> Any:
        if self.database_url is None:
            raise RuntimeError("PostgreSQL database URL is not configured")
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("psycopg is not installed") from exc
        return psycopg.connect(_normalize_database_url(self.database_url))

    def _initialize_database(self) -> None:
        if self.database_path is not None:
            self._initialize_sqlite_database()
            return
        self._initialize_postgres_database()

    def _initialize_sqlite_database(self) -> None:
        with self._connect_sqlite() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnosis_records (
                    diagnosis_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    def _load_database(self) -> None:
        if self.database_path is not None:
            self._load_sqlite_database()
            return
        self._load_postgres_database()

    def _load_sqlite_database(self) -> None:
        with self._connect_sqlite() as connection:
            for payload, in connection.execute("SELECT payload FROM diagnosis_records"):
                record = _record_from_json(payload)
                self.records[record.diagnosis_id] = record

    def _save_record(self, record: TraceRecord) -> None:
        if self.database_path is not None:
            self._save_sqlite_record(record)
            return
        self._save_postgres_record(record)

    def _save_sqlite_record(self, record: TraceRecord) -> None:
        with self._connect_sqlite() as connection:
            connection.execute(
                """
                INSERT INTO diagnosis_records (diagnosis_id, created_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(diagnosis_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (record.diagnosis_id, record.created_at, _record_to_json(record)),
            )

    def _initialize_postgres_database(self) -> None:
        with self._connect_postgres() as connection:
            connection.execute(f"CREATE SCHEMA IF NOT EXISTS {DIAGNOSIS_SCHEMA}")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {DIAGNOSIS_SCHEMA}.records (
                    diagnosis_id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    payload JSONB NOT NULL
                )
                """
            )
            connection.commit()

    def _ensure_record_loaded(self, diagnosis_id: str) -> None:
        if diagnosis_id in self.records:
            return
        if self.database_path is not None:
            self._load_sqlite_record(diagnosis_id)
            return
        self._load_postgres_record(diagnosis_id)

    def _load_sqlite_record(self, diagnosis_id: str) -> None:
        with self._connect_sqlite() as connection:
            row = connection.execute(
                "SELECT payload FROM diagnosis_records WHERE diagnosis_id = ?",
                (diagnosis_id,),
            ).fetchone()
            if row is None:
                return
            self.records[diagnosis_id] = _record_from_json(row[0])

    def _load_postgres_record(self, diagnosis_id: str) -> None:
        with self._connect_postgres() as connection:
            row = connection.execute(
                f"SELECT payload FROM {DIAGNOSIS_SCHEMA}.records WHERE diagnosis_id = %s",
                (diagnosis_id,),
            ).fetchone()
            if row is None:
                return
            self.records[diagnosis_id] = _record_from_payload(row[0])

    def _load_postgres_database(self) -> None:
        with self._connect_postgres() as connection:
            for payload, in connection.execute(f"SELECT payload FROM {DIAGNOSIS_SCHEMA}.records"):
                record = _record_from_payload(payload)
                self.records[record.diagnosis_id] = record

    def _save_postgres_record(self, record: TraceRecord) -> None:
        with self._connect_postgres() as connection:
            connection.execute(
                f"""
                INSERT INTO {DIAGNOSIS_SCHEMA}.records (diagnosis_id, created_at, payload)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT(diagnosis_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (record.diagnosis_id, record.created_at, _record_to_json(record)),
            )
            connection.commit()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _database_url_from_env() -> str:
    load_project_env()
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def _normalize_database_url(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def _record_to_json(record: TraceRecord) -> str:
    return json.dumps(_record_payload(record), ensure_ascii=False, separators=(",", ":"))


def _record_payload(record: TraceRecord) -> dict[str, object]:
    return {
        "diagnosis_id": record.diagnosis_id,
        "trace_id": record.trace_id,
        "route": record.route,
        "status": record.status,
        "diagnosis": record.diagnosis,
        "risk_level": record.risk_level,
        "confidence": record.confidence,
        "reason": record.reason,
        "requires_human_review": record.requires_human_review,
        "created_at": record.created_at,
        "steps": list(record.steps),
        "summary": record.summary,
        "events": list(record.events),
    }


def _record_from_json(payload: str) -> TraceRecord:
    return _record_from_payload(json.loads(payload))


def _record_from_payload(payload: object) -> TraceRecord:
    data = payload if isinstance(payload, dict) else json.loads(str(payload))
    return _record_from_mapping(data)


def _record_from_mapping(data: dict[str, object]) -> TraceRecord:
    return TraceRecord(
        diagnosis_id=str(data["diagnosis_id"]),
        trace_id=str(data["trace_id"]),
        route=data["route"],
        status=data["status"],
        diagnosis=_optional_string(data.get("diagnosis")),
        risk_level=_optional_string(data.get("risk_level")),
        confidence=_optional_float(data.get("confidence")),
        reason=str(data["reason"]),
        requires_human_review=bool(data["requires_human_review"]),
        created_at=str(data["created_at"]),
        steps=tuple(str(step) for step in data.get("steps", [])),
        summary={str(key): str(value) for key, value in data.get("summary", {}).items()},
        events=tuple(_event_payload(event) for event in data.get("events", [])),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _event_payload(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return {}


def _json_value(value: Any) -> object:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


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


def _timeline_for_record(record: TraceRecord) -> list[CaseTimelineEvent]:
    events = [
        CaseTimelineEvent(
            kind="diagnosis",
            title="진단 생성",
            body=f"{_route_label(record.route)} 경로가 {_status_label(record.status)} 상태를 반환했습니다.",
            created_at=record.created_at,
        )
    ]
    events.extend(_trace_timeline_events(record))
    return sorted(events, key=lambda event: event.created_at)


def _trace_timeline_events(record: TraceRecord) -> list[CaseTimelineEvent]:
    return [
        CaseTimelineEvent(
            kind="trace",
            title=_step_label(str(event.get("name", "trace event"))),
            body=_event_kind_label(str(event.get("kind", "agent"))),
            created_at=_trace_event_created_at(event, record.created_at),
        )
        for event in record.events
    ]


def _trace_event_created_at(event: dict[str, object], fallback: str) -> str:
    created_at = event.get("created_at")
    return created_at if isinstance(created_at, str) and created_at else fallback


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
        "router": "라우터",
        "tool": "모델/도구",
    }
    return labels.get(kind, kind)


trace_store = TraceStore()
