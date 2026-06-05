from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from service.backend.app.schemas import DiagnosisResponse, TraceResponse
from service.backend.app.store import trace_store
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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent_mode": "mock_inference"}


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    prpd_image: Annotated[UploadFile | None, File()] = None,
    timeseries_csv: Annotated[UploadFile | None, File()] = None,
    metadata: Annotated[str | None, Form()] = None,
) -> DiagnosisResponse:
    metadata_input, metadata_error = parse_metadata(metadata)
    csv_shape = None
    if timeseries_csv is not None:
        csv_shape = inspect_csv_shape(await timeseries_csv.read())
    has_png_image = False
    if prpd_image is not None:
        has_png_image = is_png_upload(prpd_image.filename, prpd_image.content_type)
    return run_diagnosis_workflow(
        WorkflowInput(
            has_png_image=has_png_image,
            csv_shape=csv_shape,
            metadata=metadata_input,
            metadata_error=metadata_error,
        )
    )


@app.get("/diagnose/{diagnosis_id}/trace", response_model=TraceResponse)
def diagnosis_trace(diagnosis_id: str) -> TraceResponse:
    trace = trace_store.get(diagnosis_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="diagnosis trace not found")
    return trace
