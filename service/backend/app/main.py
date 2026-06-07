from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from service.backend.app.artifacts import store_upload
from service.backend.app.openai_agents_adapter import check_agents_sdk
from service.backend.app.schemas import DiagnosisListResponse, DiagnosisResponse, ModelRuntimeStatus, TraceResponse
from service.backend.app.store import trace_store
from service.backend.app.tools import time_series_adapter, vlm_adapter
from service.backend.app.validation import inspect_csv_shape, is_png_upload
from service.backend.app.workflow import WorkflowInput, parse_metadata, run_diagnosis_workflow

app = FastAPI(title="Partial Discharge Diagnosis Agent Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str | bool]:
    agents_sdk = check_agents_sdk()
    return {
        "status": "ok",
        "agent_mode": "local_deterministic",
        "agents_sdk_installed": agents_sdk.installed,
        "agents_sdk_reason": agents_sdk.reason,
    }


@app.get("/model-status", response_model=ModelRuntimeStatus)
def model_status() -> ModelRuntimeStatus:
    agents_sdk = check_agents_sdk()
    return ModelRuntimeStatus(
        agent_mode="local_deterministic",
        agents_sdk_installed=agents_sdk.installed,
        agents_sdk_reason=agents_sdk.reason,
        time_series_model=time_series_adapter.model_name,
        time_series_version=time_series_adapter.model_version,
        vlm_model=vlm_adapter.model_name,
        vlm_version=vlm_adapter.model_version,
    )


@app.get("/diagnoses", response_model=DiagnosisListResponse)
def diagnoses(limit: int = 20) -> DiagnosisListResponse:
    return DiagnosisListResponse(items=trace_store.list(limit=limit))


@app.get("/review-queue", response_model=DiagnosisListResponse)
def review_queue(limit: int = 20) -> DiagnosisListResponse:
    return DiagnosisListResponse(items=trace_store.review_queue(limit=limit))


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    prpd_image: Annotated[UploadFile | None, File()] = None,
    timeseries_csv: Annotated[UploadFile | None, File()] = None,
    metadata: Annotated[str | None, Form()] = None,
) -> DiagnosisResponse:
    diagnosis_id = f"diag_{uuid4().hex[:12]}"
    metadata_input, metadata_error = parse_metadata(metadata)
    csv_shape = None
    csv_artifact = None
    if timeseries_csv is not None:
        csv_content = await timeseries_csv.read()
        csv_shape = inspect_csv_shape(csv_content)
        csv_artifact = store_upload(
            csv_content,
            diagnosis_id=diagnosis_id,
            filename=timeseries_csv.filename or "signal.csv",
            content_type=timeseries_csv.content_type,
        )
    has_png_image = False
    image_artifact = None
    if prpd_image is not None:
        has_png_image = is_png_upload(prpd_image.filename, prpd_image.content_type)
        image_content = await prpd_image.read()
        image_artifact = store_upload(
            image_content,
            diagnosis_id=diagnosis_id,
            filename=prpd_image.filename or "prpd.png",
            content_type=prpd_image.content_type,
        )
    return run_diagnosis_workflow(
        WorkflowInput(
            diagnosis_id=diagnosis_id,
            has_png_image=has_png_image,
            csv_shape=csv_shape,
            metadata=metadata_input,
            metadata_error=metadata_error,
            image_path=image_artifact.path if image_artifact is not None else None,
            image_sha256=image_artifact.sha256 if image_artifact is not None else None,
            csv_path=csv_artifact.path if csv_artifact is not None else None,
            csv_sha256=csv_artifact.sha256 if csv_artifact is not None else None,
        )
    )


@app.get("/diagnose/{diagnosis_id}/trace", response_model=TraceResponse)
def diagnosis_trace(diagnosis_id: str) -> TraceResponse:
    trace = trace_store.get(diagnosis_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="diagnosis trace not found")
    return trace
