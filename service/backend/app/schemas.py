from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DiagnosisRoute = Literal["insufficient_input", "timeseries_only", "vlm_only", "hybrid"]
DiagnosisStatus = Literal["completed", "needs_review", "rejected"]
EvidenceSource = Literal["time_series", "vision", "vlm", "rag", "similar_case"]
AgreementLevel = Literal["none", "single_source", "agreement", "partial_agreement", "conflict"]


class MetadataInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    equipment_name: str = Field(min_length=1)
    equipment_type: str | None = None
    equipment_rated_voltage: str = Field(min_length=1)
    equipment_rated_current: str = Field(min_length=1)
    sensor_type: str = Field(min_length=1)
    measurement_location: str | None = None
    operating_condition: str | None = None
    temperature: float
    humidity: float
    insulator_type: str | None = None
    insulator_name: str | None = None
    clearance_distance: str | None = None


class EvidenceFactor(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    value: str | int | float | None
    weight: float
    explanation: str


class StandardModelEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: EvidenceSource
    model_name: str
    model_version: str = "unknown"
    label_id: int | None = None
    label_name: str | None = None
    confidence: float | None = None
    uncertainty: float | None = None
    ood_score: float | None = None
    top_factors: list[EvidenceFactor] = Field(default_factory=list)
    explanation: str


class TimeSeriesResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    model_version: str = "unknown"
    label_id: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    features: dict[str, float]
    standard_evidence: StandardModelEvidence | None = None


class VisionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    model_version: str = "unknown"
    label_id: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    evidence: dict[str, float | str]
    standard_evidence: StandardModelEvidence | None = None


class RagDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    source: str
    excerpt: str
    relevance: float
    source_type: str | None = None
    metadata: dict[str, str | int | float | None] = Field(default_factory=dict)


class SimilarCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    label_id: int
    label_name: str
    equipment_name: str
    insulator_type: str
    sensor_type: str
    clearance_distance: str
    similarity: float
    reason: str
    image_url: str
    metadata: dict[str, str | int | float | None]


class SimilarCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    retriever_name: str
    retriever_version: str = "unknown"
    query: str
    cases: list[SimilarCase]


class DatasetCaseListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[SimilarCase]


class RagResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    retriever_name: str
    retriever_version: str = "unknown"
    query: str
    documents: list[RagDocument]
    similar_cases: list[SimilarCase] = Field(default_factory=list)


class RagSourceCount(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents: int
    chunks: int


class RagStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    ready: bool
    database_name: str
    vector_extension: str | None = None
    embedding_model: str
    vector_dim: int
    top_k: int
    source_types: list[str]
    document_count: int
    chunk_count: int
    query_log_count: int
    source_counts: dict[str, RagSourceCount] = Field(default_factory=dict)
    error: str | None = None


class RagDocumentListItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_key: str
    source_type: str
    title: str
    source_path: str | None = None
    updated_at: str
    chunk_count: int


class RagDocumentListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[RagDocumentListItem]
    error: str | None = None


class RagQueryLogItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    diagnosis_id: str | None = None
    query_text: str
    query_metadata: dict[str, object] = Field(default_factory=dict)
    retrieved_chunks: list[object] = Field(default_factory=list)
    created_at: str


class RagQueryLogResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[RagQueryLogItem]
    error: str | None = None


class RagSearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)


class RagSearchResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    documents: list[RagDocument]
    error: str | None = None


class RagReindexRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_limit: int | None = Field(default=None, ge=1, le=50000)


class RagReindexResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_count: int
    chunk_count: int
    dataset_limit: int | None = None
    embedding_model: str


class VlmResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    model_version: str = "unknown"
    label_id: int
    diagnosis: str
    risk_level: str
    confidence: float
    reason: str
    recommended_action: str
    standard_evidence: StandardModelEvidence | None = None


class FusionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: str
    final_label_id: int | None = None
    final_label_name: str | None = None
    confidence: float | None = None
    agreement_level: AgreementLevel
    contributing_sources: list[EvidenceSource] = Field(default_factory=list)
    rationale: str
    evidence: list[StandardModelEvidence] = Field(default_factory=list)


class DiagnosisResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    diagnosis_id: str
    route: DiagnosisRoute
    status: DiagnosisStatus
    final_label_id: int | None = None
    diagnosis: str | None = None
    risk_level: str | None = None
    confidence: float | None = None
    reason: str
    recommended_action: str | None = None
    requires_human_review: bool
    trace_id: str
    error_code: str | None = None


class DiagnosisListItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    diagnosis_id: str
    trace_id: str
    route: DiagnosisRoute
    status: DiagnosisStatus
    diagnosis: str | None = None
    risk_level: str | None = None
    confidence: float | None = None
    reason: str
    requires_human_review: bool
    created_at: str


class DiagnosisListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[DiagnosisListItem]


class ReviewActionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["approve", "request_retest", "dispatch_field_team", "mark_false_positive"]
    note: str = Field(default="", max_length=1000)


class ReviewActionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: str
    note: str
    created_at: str


class DiagnosisCommentRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    note: str = Field(min_length=1, max_length=2000)


class DiagnosisCommentRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    note: str
    created_at: str


class CaseTimelineEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["diagnosis", "trace", "action", "comment"]
    title: str
    body: str
    created_at: str


class DiagnosisDetailResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    diagnosis: DiagnosisListItem
    trace: "TraceResponse"
    actions: list[ReviewActionRecord] = Field(default_factory=list)
    comments: list[DiagnosisCommentRecord] = Field(default_factory=list)
    timeline: list[CaseTimelineEvent] = Field(default_factory=list)


class DiagnosisReportResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    detail: DiagnosisDetailResponse
    reference_cases: list[SimilarCase] = Field(default_factory=list)


class DemoScenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario_id: str
    title: str
    diagnosis_id: str
    route: DiagnosisRoute
    status: DiagnosisStatus
    risk_level: str | None = None
    summary: str


class DemoScenarioListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenarios: list[DemoScenario]


class DemoSeedResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    seeded: list[str]
    scenarios: list[DemoScenario]


class DemoScenarioActivationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario: DemoScenario
    detail: DiagnosisDetailResponse


class ModelRuntimeStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_mode: str
    agents_sdk_installed: bool
    agents_sdk_reason: str
    adapter_mode: str
    artifact_root: str
    time_series_model: str
    time_series_version: str
    time_series_adapter: str
    time_series_ready: bool
    time_series_manifest: str | None = None
    time_series_checkpoint: str | None = None
    time_series_error: str | None = None
    vision_model: str
    vision_version: str
    vision_adapter: str
    vision_ready: bool
    vision_manifest: str | None = None
    vision_checkpoint: str | None = None
    vision_error: str | None = None
    case_retriever: str
    case_version: str
    rag_retriever: str
    rag_version: str
    vlm_model: str
    vlm_version: str
    vlm_adapter: str
    vlm_ready: bool
    vlm_manifest: str | None = None
    vlm_checkpoint: str | None = None
    vlm_error: str | None = None
    llm_rag_provider: str
    llm_rag_adapter: str
    llm_rag_ready: bool
    llm_rag_model: str | None = None
    llm_rag_error: str | None = None


class TraceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    diagnosis_id: str
    trace_id: str
    route: DiagnosisRoute
    status: DiagnosisStatus
    steps: list[str]
    summary: dict[str, str]
    events: list[dict[str, object]] = Field(default_factory=list)
