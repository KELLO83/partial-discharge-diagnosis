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
