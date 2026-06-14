from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient

from service.backend.app.main import app


def _csv_payload(rows: int = 20, cols: int = 7680) -> bytes:
    row = ",".join("0" for _ in range(cols))
    return ("\n".join(row for _ in range(rows)) + "\n").encode("utf-8")


def test_diagnose_routes_to_timeseries_when_only_csv_is_provided() -> None:
    client = TestClient(app)

    response = client.post(
        "/diagnose",
        files={"timeseries_csv": ("signal.csv", io.BytesIO(_csv_payload()), "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "timeseries_only"
    assert payload["status"] == "completed"
    assert payload["final_label_id"] == 3
    assert payload["diagnosis_id"].startswith("diag_")


def test_diagnose_routes_to_vlm_when_image_and_metadata_are_provided() -> None:
    client = TestClient(app)
    metadata = {
        "equipment_name": "ACSR-OC",
        "equipment_rated_voltage": "22900V",
        "equipment_rated_current": "268A",
        "sensor_type": "HFCT",
        "temperature": 19,
        "humidity": 66,
    }

    response = client.post(
        "/diagnose",
        data={"metadata": json.dumps(metadata, ensure_ascii=False)},
        files={"prpd_image": ("prpd.png", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "vlm_only"
    assert payload["status"] == "completed"
    assert payload["diagnosis"] == "코로나 방전"


def test_diagnose_rejects_invalid_csv_shape() -> None:
    client = TestClient(app)

    response = client.post(
        "/diagnose",
        files={"timeseries_csv": ("signal.csv", io.BytesIO(_csv_payload(rows=2, cols=3)), "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "insufficient_input"
    assert payload["status"] == "rejected"
    assert payload["error_code"] == "INVALID_INPUT"


def test_diagnose_rejects_label_leaking_metadata() -> None:
    client = TestClient(app)
    metadata = {
        "equipment_name": "ACSR-OC",
        "equipment_rated_voltage": "22900V",
        "equipment_rated_current": "268A",
        "sensor_type": "HFCT",
        "temperature": 19,
        "humidity": 66,
        "label_id": 3,
    }

    response = client.post(
        "/diagnose",
        data={"metadata": json.dumps(metadata, ensure_ascii=False)},
        files={"prpd_image": ("prpd.png", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "rejected"
    assert payload["error_code"] == "INVALID_INPUT"


def test_trace_endpoint_returns_agent_steps_after_diagnosis() -> None:
    client = TestClient(app)
    response = client.post(
        "/diagnose",
        files={"timeseries_csv": ("signal.csv", io.BytesIO(_csv_payload()), "text/csv")},
    )
    diagnosis_id = response.json()["diagnosis_id"]

    trace_response = client.get(f"/diagnose/{diagnosis_id}/trace")

    assert trace_response.status_code == 200
    trace = trace_response.json()
    assert trace["diagnosis_id"] == diagnosis_id
    assert "input_router" in trace["steps"]
    assert "input_artifacts" in trace["steps"]
    assert "time_series_tool" in trace["steps"]
    artifact_event = next(event for event in trace["events"] if event["name"] == "input_artifacts")
    time_series_event = next(event for event in trace["events"] if event["name"] == "time_series_tool")
    assert artifact_event["summary"]["timeseries_csv_url"].endswith("/artifacts/timeseries-csv")
    assert artifact_event["summary"]["timeseries_signal"]["sample_count"] == 153600
    assert time_series_event["summary"]["csv_sha256"]
    assert "features" in time_series_event["summary"]


def test_hybrid_diagnosis_runs_time_series_vision_and_vlm_tools() -> None:
    client = TestClient(app)
    metadata = {
        "equipment_name": "ACSR-OC",
        "equipment_rated_voltage": "22900V",
        "equipment_rated_current": "268A",
        "sensor_type": "HFCT",
        "temperature": 19,
        "humidity": 66,
    }

    response = client.post(
        "/diagnose",
        data={"metadata": json.dumps(metadata, ensure_ascii=False)},
        files={
            "timeseries_csv": ("signal.csv", io.BytesIO(_csv_payload()), "text/csv"),
            "prpd_image": ("prpd.png", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png"),
        },
    )
    diagnosis_id = response.json()["diagnosis_id"]
    trace_response = client.get(f"/diagnose/{diagnosis_id}/trace")
    report_response = client.get(f"/diagnoses/{diagnosis_id}/report")

    assert response.status_code == 200
    assert trace_response.status_code == 200
    assert report_response.status_code == 200
    trace_steps = trace_response.json()["steps"]
    assert "metadata_context" in trace_steps
    assert "input_artifacts" in trace_steps
    assert "time_series_tool" in trace_steps
    assert "vision_tool" in trace_steps
    assert "similar_case_tool" in trace_steps
    assert "rag_tool" in trace_steps
    assert "vlm_tool" in trace_steps
    assert "fusion_engine" in trace_steps
    trace = trace_response.json()
    artifact_event = next(event for event in trace["events"] if event["name"] == "input_artifacts")
    similar_event = next(event for event in trace["events"] if event["name"] == "similar_case_tool")
    fusion_event = next(event for event in trace["events"] if event["name"] == "fusion_engine")
    image_response = client.get(artifact_event["summary"]["prpd_image_url"])
    csv_response = client.get(artifact_event["summary"]["timeseries_csv_url"])
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert similar_event["summary"]["case_count"] == 5
    assert similar_event["summary"]["cases"][0]["image_url"].startswith("/dataset/cases/")
    assert fusion_event["summary"]["agreement_level"] in {"agreement", "partial_agreement", "conflict"}
    assert fusion_event["summary"]["evidence"][0]["top_factors"]
    assert len(report_response.json()["reference_cases"]) == 5


def test_health_reports_agent_runtime_mode() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_mode"] == "local_deterministic"
    assert "agents_sdk_installed" in payload


def test_diagnosis_history_returns_recent_diagnoses() -> None:
    client = TestClient(app)
    diagnosis = client.post(
        "/diagnose",
        files={"timeseries_csv": ("signal.csv", io.BytesIO(_csv_payload()), "text/csv")},
    ).json()

    response = client.get("/diagnoses")

    assert response.status_code == 200
    items = response.json()["items"]
    assert any(item["diagnosis_id"] == diagnosis["diagnosis_id"] for item in items)
    assert items[0]["created_at"]


def test_review_queue_returns_human_review_items() -> None:
    client = TestClient(app)
    response = client.post(
        "/diagnose",
        files={"timeseries_csv": ("bad.csv", io.BytesIO(_csv_payload(rows=2, cols=3)), "text/csv")},
    )

    queue_response = client.get("/review-queue")

    assert response.status_code == 200
    assert queue_response.status_code == 200
    queue = queue_response.json()["items"]
    assert any(item["status"] == "rejected" for item in queue)
    assert all(item["requires_human_review"] or item["status"] in {"needs_review", "rejected"} for item in queue)


def test_model_status_reports_adapter_versions() -> None:
    client = TestClient(app)

    response = client.get("/model-status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["time_series_model"] == "mock_patchtst"
    assert payload["adapter_mode"] == "mock"
    assert payload["time_series_adapter"] == "mock"
    assert payload["vision_adapter"] == "mock"
    assert payload["vision_model"] == "mock_prpd_small_cnn"
    assert payload["case_retriever"] == "domain_feature_case_retriever"
    assert payload["rag_retriever"] == "pgvector_rulebook_case_rag"
    assert payload["vlm_model"] == "mock_qwen3_vl_2b"
    assert payload["vlm_adapter"] == "mock"
    assert payload["llm_rag_provider"] == "mock"
    assert payload["llm_rag_adapter"] == "mock_qwen3_vl_2b"
    assert payload["llm_rag_ready"] is False


def test_rag_admin_endpoints_return_operational_state() -> None:
    client = TestClient(app)

    status_response = client.get("/rag/status")
    documents_response = client.get("/rag/documents?limit=3")
    logs_response = client.get("/rag/query-logs?limit=3")
    search_response = client.post("/rag/search", json={"query": "HFCT 코로나 방전 근거", "top_k": 2})

    assert status_response.status_code == 200
    assert documents_response.status_code == 200
    assert logs_response.status_code == 200
    assert search_response.status_code == 200
    status = status_response.json()
    assert status["embedding_model"] == "dragonkue/multilingual-e5-small-ko-v2"
    assert "ready" in status
    assert "source_counts" in status
    assert "items" in documents_response.json()
    assert "items" in logs_response.json()
    assert "documents" in search_response.json()


def test_diagnosis_detail_actions_comments_and_report() -> None:
    client = TestClient(app)
    diagnosis = client.post(
        "/diagnose",
        files={"timeseries_csv": ("signal.csv", io.BytesIO(_csv_payload()), "text/csv")},
    ).json()
    diagnosis_id = diagnosis["diagnosis_id"]

    action_response = client.post(
        f"/diagnoses/{diagnosis_id}/actions",
        json={"action": "approve", "note": "confirmed by operator"},
    )
    comment_response = client.post(
        f"/diagnoses/{diagnosis_id}/comments",
        json={"note": "maintenance ticket created"},
    )
    detail_response = client.get(f"/diagnoses/{diagnosis_id}")
    report_response = client.get(f"/diagnoses/{diagnosis_id}/report")

    assert action_response.status_code == 200
    assert comment_response.status_code == 200
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["diagnosis"]["diagnosis_id"] == diagnosis_id
    assert detail["actions"][0]["action"] == "approve"
    assert detail["comments"][0]["note"] == "maintenance ticket created"
    assert report_response.status_code == 200
    assert report_response.json()["detail"]["trace"]["diagnosis_id"] == diagnosis_id


def test_dataset_case_endpoints_expose_reference_images() -> None:
    client = TestClient(app)

    cases_response = client.get("/dataset/cases?limit=1")

    assert cases_response.status_code == 200
    case = cases_response.json()["items"][0]
    image_response = client.get(case["image_url"])
    detail_response = client.get(f"/dataset/cases/{case['sample_id']}")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"
    assert detail_response.status_code == 200
    assert detail_response.json()["sample_id"] == case["sample_id"]


def test_dataset_case_search_filters_by_label_and_sensor() -> None:
    client = TestClient(app)

    response = client.get("/dataset/cases/search?label_id=1&sensor_type=HFCT&limit=3")

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 3
    assert all(item["label_id"] == 1 for item in items)
    assert all(item["sensor_type"] == "HFCT" for item in items)


def test_demo_seed_and_scenario_activation_populates_detail_timeline() -> None:
    client = TestClient(app)

    scenarios_response = client.get("/demo/scenarios")
    seed_response = client.post("/demo/seed")
    activate_response = client.post("/demo/scenarios/model_disagreement/activate")

    assert scenarios_response.status_code == 200
    assert len(scenarios_response.json()["scenarios"]) == 5
    assert seed_response.status_code == 200
    assert "demo_disagree_0001" in seed_response.json()["seeded"]
    assert activate_response.status_code == 200
    payload = activate_response.json()
    assert payload["scenario"]["scenario_id"] == "model_disagreement"
    assert payload["detail"]["diagnosis"]["status"] == "needs_review"
    assert payload["detail"]["timeline"]
    assert any(event["kind"] == "action" for event in payload["detail"]["timeline"])
