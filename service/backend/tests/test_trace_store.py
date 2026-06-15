from __future__ import annotations

from service.backend.app.infrastructure.store import TraceRecord, TraceStore


def test_trace_store_persists_records_actions_and_comments(tmp_path) -> None:
    database_path = tmp_path / "diagnosis_history.sqlite3"
    store = TraceStore(database_path=database_path)
    store.save(persisted_record())
    assert store.add_action("diag_persisted", "request_retest", "재측정") is not None
    assert store.add_comment("diag_persisted", "운영 메모") is not None

    restored = TraceStore(database_path=database_path)
    detail = restored.detail("diag_persisted")

    assert detail is not None
    assert detail.diagnosis.diagnosis_id == "diag_persisted"
    assert detail.trace.steps == ["input_router", "vlm_tool"]
    assert detail.actions[0].note == "재측정"
    assert detail.comments[0].note == "운영 메모"
    assert restored.review_queue()[0].diagnosis_id == "diag_persisted"


def test_trace_store_loads_record_saved_by_another_instance(tmp_path) -> None:
    database_path = tmp_path / "diagnosis_history.sqlite3"
    reader = TraceStore(database_path=database_path)
    writer = TraceStore(database_path=database_path)

    writer.save(persisted_record())

    detail = reader.detail("diag_persisted")

    assert detail is not None
    assert detail.diagnosis.diagnosis_id == "diag_persisted"
    assert detail.trace.steps == ["input_router", "vlm_tool"]


def persisted_record() -> TraceRecord:
    return TraceRecord(
        diagnosis_id="diag_persisted",
        trace_id="trace_persisted",
        route="hybrid",
        status="needs_review",
        diagnosis="코로나 방전",
        risk_level="주의",
        confidence=0.82,
        reason="검토 필요",
        requires_human_review=True,
        created_at="2026-06-15T00:00:00+00:00",
        steps=("input_router", "vlm_tool"),
        summary={"reason": "검토 필요"},
        events=({"name": "input_router", "kind": "guardrail", "summary": {"route": "hybrid"}},),
    )
