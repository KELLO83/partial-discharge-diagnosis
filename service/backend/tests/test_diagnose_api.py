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
    assert "time_series_tool" in trace["steps"]
    assert trace["events"][1]["summary"]["csv_sha256"]


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
    assert payload["vlm_model"] == "mock_qwen3_vl_2b"
