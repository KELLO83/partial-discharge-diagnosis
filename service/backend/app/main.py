from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from service.backend.app.infrastructure.artifacts import find_uploaded_file, store_upload
from service.backend.app.application.demo import (
    activate_demo_scenario,
    demo_scenarios,
    seed_demo_records,
)
from service.backend.app.infrastructure.openai_agents_adapter import check_agents_sdk
from service.backend.app.rag.admin import (
    list_rag_documents,
    list_rag_query_logs,
    read_rag_status,
    reindex_rag_documents,
    search_rag_documents,
)
from service.backend.app.schemas import (
    DemoScenarioActivationResponse,
    DemoScenarioListResponse,
    DemoSeedResponse,
    DatasetCaseListResponse,
    DiagnosisCommentRecord,
    DiagnosisCommentRequest,
    DiagnosisDetailResponse,
    DiagnosisListResponse,
    DiagnosisReportResponse,
    DiagnosisResponse,
    ModelRuntimeStatus,
    RagDocumentListResponse,
    RagQueryLogResponse,
    RagReindexRequest,
    RagReindexResponse,
    RagSearchRequest,
    RagSearchResponse,
    RagStatusResponse,
    ReviewActionRecord,
    ReviewActionRequest,
    SimilarCase,
    TraceResponse,
)
from service.backend.app.domain.similar_cases import dataset_case_repository, to_similar_case
from service.backend.app.infrastructure.store import trace_store
from service.backend.app.application.adapters import (
    llm_rag_status,
    model_runtime,
    rag_adapter,
    similar_case_adapter,
    time_series_adapter,
    vision_adapter,
    vlm_adapter,
)
from service.backend.app.infrastructure.validation import inspect_csv_shape, is_png_upload, summarize_csv_signal
from service.backend.app.application.workflow import WorkflowInput, parse_metadata, run_diagnosis_workflow

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
    time_series_info = model_runtime.info_for("time_series")
    vision_info = model_runtime.info_for("vision")
    vlm_info = model_runtime.info_for("vlm")
    return ModelRuntimeStatus(
        agent_mode="local_deterministic",
        agents_sdk_installed=agents_sdk.installed,
        agents_sdk_reason=agents_sdk.reason,
        adapter_mode=model_runtime.mode,
        artifact_root=str(model_runtime.artifact_root),
        time_series_model=time_series_adapter.model_name,
        time_series_version=time_series_adapter.model_version,
        time_series_adapter=time_series_info.adapter_kind,
        time_series_ready=time_series_info.ready,
        time_series_manifest=time_series_info.manifest_path,
        time_series_checkpoint=time_series_info.checkpoint_path,
        time_series_error=time_series_info.error,
        vision_model=vision_adapter.model_name,
        vision_version=vision_adapter.model_version,
        vision_adapter=vision_info.adapter_kind,
        vision_ready=vision_info.ready,
        vision_manifest=vision_info.manifest_path,
        vision_checkpoint=vision_info.checkpoint_path,
        vision_error=vision_info.error,
        case_retriever=similar_case_adapter.model_name,
        case_version=similar_case_adapter.model_version,
        rag_retriever=rag_adapter.model_name,
        rag_version=rag_adapter.model_version,
        vlm_model=vlm_adapter.model_name,
        vlm_version=vlm_adapter.model_version,
        vlm_adapter=vlm_info.adapter_kind,
        vlm_ready=vlm_info.ready,
        vlm_manifest=vlm_info.manifest_path,
        vlm_checkpoint=vlm_info.checkpoint_path,
        vlm_error=vlm_info.error,
        llm_rag_provider=llm_rag_status.provider,
        llm_rag_adapter=llm_rag_status.active_adapter,
        llm_rag_ready=llm_rag_status.ready,
        llm_rag_model=llm_rag_status.model,
        llm_rag_error=llm_rag_status.error,
    )


@app.get("/rag/status", response_model=RagStatusResponse)
def rag_status() -> RagStatusResponse:
    return read_rag_status()


@app.get("/rag/documents", response_model=RagDocumentListResponse)
def rag_documents(source_type: str | None = None, limit: int = 50) -> RagDocumentListResponse:
    return list_rag_documents(source_type=source_type, limit=limit)


@app.get("/rag/query-logs", response_model=RagQueryLogResponse)
def rag_query_logs(limit: int = 20) -> RagQueryLogResponse:
    return list_rag_query_logs(limit=limit)


@app.post("/rag/search", response_model=RagSearchResponse)
def rag_search(request: RagSearchRequest) -> RagSearchResponse:
    return search_rag_documents(query=request.query, top_k=request.top_k)


@app.post("/rag/reindex", response_model=RagReindexResponse)
def rag_reindex(request: RagReindexRequest) -> RagReindexResponse:
    try:
        return reindex_rag_documents(dataset_limit=request.dataset_limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/diagnoses", response_model=DiagnosisListResponse)
def diagnoses(limit: int = 20) -> DiagnosisListResponse:
    return DiagnosisListResponse(items=trace_store.list(limit=limit))


@app.get("/review-queue", response_model=DiagnosisListResponse)
def review_queue(limit: int = 20) -> DiagnosisListResponse:
    return DiagnosisListResponse(items=trace_store.review_queue(limit=limit))


@app.get("/diagnoses/{diagnosis_id}", response_model=DiagnosisDetailResponse)
def diagnosis_detail(diagnosis_id: str) -> DiagnosisDetailResponse:
    detail = trace_store.detail(diagnosis_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="diagnosis detail not found")
    return detail


@app.post("/diagnoses/{diagnosis_id}/actions", response_model=ReviewActionRecord)
def add_review_action(diagnosis_id: str, request: ReviewActionRequest) -> ReviewActionRecord:
    record = trace_store.add_action(diagnosis_id, request.action, request.note)
    if record is None:
        raise HTTPException(status_code=404, detail="diagnosis not found")
    return record


@app.post("/diagnoses/{diagnosis_id}/comments", response_model=DiagnosisCommentRecord)
def add_diagnosis_comment(diagnosis_id: str, request: DiagnosisCommentRequest) -> DiagnosisCommentRecord:
    record = trace_store.add_comment(diagnosis_id, request.note)
    if record is None:
        raise HTTPException(status_code=404, detail="diagnosis not found")
    return record


@app.get("/diagnoses/{diagnosis_id}/report", response_model=DiagnosisReportResponse)
def diagnosis_report(diagnosis_id: str) -> DiagnosisReportResponse:
    detail = trace_store.detail(diagnosis_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="diagnosis report not found")
    return DiagnosisReportResponse(detail=detail, reference_cases=_reference_cases_from_trace(detail.trace))


@app.get("/diagnoses/{diagnosis_id}/artifacts/prpd-image")
def diagnosis_prpd_image(diagnosis_id: str) -> FileResponse:
    image_path = find_uploaded_file(diagnosis_id, {".png"})
    if image_path is None:
        raise HTTPException(status_code=404, detail="diagnosis image not found")
    return FileResponse(image_path, media_type="image/png")


@app.get("/diagnoses/{diagnosis_id}/artifacts/timeseries-csv")
def diagnosis_timeseries_csv(diagnosis_id: str) -> FileResponse:
    csv_path = find_uploaded_file(diagnosis_id, {".csv"})
    if csv_path is None:
        raise HTTPException(status_code=404, detail="diagnosis timeseries csv not found")
    return FileResponse(csv_path, media_type="text/csv")


@app.get("/dataset/cases", response_model=DatasetCaseListResponse)
def dataset_cases(limit: int = 20) -> DatasetCaseListResponse:
    cases = [
        to_similar_case(case, 1.0, "데이터셋 등록 사례")
        for case in dataset_case_repository.list(limit=limit)
    ]
    return DatasetCaseListResponse(items=cases)


@app.get("/dataset/cases/search", response_model=DatasetCaseListResponse)
def search_dataset_cases(
    limit: int = 20,
    label_id: int | None = None,
    equipment_name: str | None = None,
    sensor_type: str | None = None,
    insulator_type: str | None = None,
    query: str | None = None,
) -> DatasetCaseListResponse:
    cases = dataset_case_repository.search(
        label_id=label_id,
        equipment_name=equipment_name,
        sensor_type=sensor_type,
        insulator_type=insulator_type,
        query=query,
        limit=limit,
    )
    return DatasetCaseListResponse(items=cases)


@app.get("/dataset/cases/{sample_id}", response_model=SimilarCase)
def dataset_case_detail(sample_id: str) -> SimilarCase:
    case = dataset_case_repository.get(sample_id)
    if case is None:
        raise HTTPException(status_code=404, detail="dataset case not found")
    return to_similar_case(case, 1.0, "데이터셋 등록 사례")


@app.get("/dataset/cases/{sample_id}/image")
def dataset_case_image(sample_id: str) -> FileResponse:
    case = dataset_case_repository.get(sample_id)
    if case is None or not case.image_path.exists():
        raise HTTPException(status_code=404, detail="dataset case image not found")
    return FileResponse(case.image_path, media_type="image/png")


@app.get("/demo/scenarios", response_model=DemoScenarioListResponse)
def list_demo_scenarios() -> DemoScenarioListResponse:
    return DemoScenarioListResponse(scenarios=demo_scenarios())


@app.post("/demo/seed", response_model=DemoSeedResponse)
def seed_demo() -> DemoSeedResponse:
    return seed_demo_records()


@app.post("/demo/scenarios/{scenario_id}/activate", response_model=DemoScenarioActivationResponse)
def activate_scenario(scenario_id: str) -> DemoScenarioActivationResponse:
    activated = activate_demo_scenario(scenario_id)
    if activated is None:
        raise HTTPException(status_code=404, detail="demo scenario not found")
    scenario, diagnosis_id = activated
    detail = trace_store.detail(diagnosis_id)
    if detail is None:
        raise HTTPException(status_code=500, detail="demo scenario detail not seeded")
    return DemoScenarioActivationResponse(scenario=scenario, detail=detail)


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    prpd_image: Annotated[UploadFile | None, File()] = None,
    timeseries_csv: Annotated[UploadFile | None, File()] = None,
    metadata: Annotated[str | None, Form()] = None,
) -> DiagnosisResponse:
    diagnosis_id = f"diag_{uuid4().hex[:12]}"
    metadata_input, metadata_error = parse_metadata(metadata)
    input_artifacts: dict[str, object] = {}
    csv_shape = None
    csv_artifact = None
    if timeseries_csv is not None:
        csv_content = await timeseries_csv.read()
        csv_shape = inspect_csv_shape(csv_content)
        input_artifacts["timeseries_signal"] = summarize_csv_signal(csv_content).to_trace_payload()
        csv_artifact = store_upload(
            csv_content,
            diagnosis_id=diagnosis_id,
            filename=timeseries_csv.filename or "signal.csv",
            content_type=timeseries_csv.content_type,
        )
        input_artifacts["timeseries_csv_url"] = f"/diagnoses/{diagnosis_id}/artifacts/timeseries-csv"
        input_artifacts["timeseries_csv_sha256"] = csv_artifact.sha256
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
        if has_png_image:
            input_artifacts["prpd_image_url"] = f"/diagnoses/{diagnosis_id}/artifacts/prpd-image"
            input_artifacts["prpd_image_sha256"] = image_artifact.sha256
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
            input_artifacts=input_artifacts,
        )
    )


@app.get("/diagnose/{diagnosis_id}/trace", response_model=TraceResponse)
def diagnosis_trace(diagnosis_id: str) -> TraceResponse:
    trace = trace_store.get(diagnosis_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="diagnosis trace not found")
    return trace


def _reference_cases_from_trace(trace: TraceResponse) -> list[SimilarCase]:
    for event in trace.events:
        if event.get("name") != "similar_case_tool":
            continue
        summary = event.get("summary", {})
        if not isinstance(summary, dict):
            return []
        cases = summary.get("cases", [])
        if not isinstance(cases, list):
            return []
        return [SimilarCase.model_validate(case) for case in cases]
    return []
