import type { InputRoute } from "./route";

export type MetadataForm = {
  readonly equipmentName: string;
  readonly equipmentType?: string;
  readonly ratedVoltage: string;
  readonly ratedCurrent: string;
  readonly sensorType: string;
  readonly measurementLocation?: string;
  readonly operatingCondition?: string;
  readonly temperature: string;
  readonly humidity: string;
  readonly insulatorType?: string;
  readonly clearanceDistance?: string;
};

export type FormPresenceInput = {
  readonly hasImage: boolean;
  readonly hasTimeseries: boolean;
  readonly metadata: MetadataForm;
};

export type DiagnosisStatus = "completed" | "needs_review" | "rejected";

export type DiagnosisResponse = {
  readonly diagnosis_id: string;
  readonly route: InputRoute;
  readonly status: DiagnosisStatus;
  readonly final_label_id: number | null;
  readonly diagnosis: string | null;
  readonly risk_level: string | null;
  readonly confidence: number | null;
  readonly reason: string;
  readonly recommended_action: string | null;
  readonly requires_human_review: boolean;
  readonly trace_id: string;
  readonly error_code: string | null;
};

export type TraceEvent = {
  readonly name: string;
  readonly kind: string;
  readonly summary: Record<string, unknown>;
  readonly created_at?: string;
};

export type TraceResponse = {
  readonly diagnosis_id: string;
  readonly trace_id: string;
  readonly route: InputRoute;
  readonly status: DiagnosisStatus;
  readonly steps: readonly string[];
  readonly summary: Record<string, string>;
  readonly events: readonly TraceEvent[];
};

export type EvidenceFactor = {
  readonly name: string;
  readonly value: string | number | null;
  readonly weight: number;
  readonly explanation: string;
};

export type StandardModelEvidence = {
  readonly source: "time_series" | "vision" | "vlm" | "rag" | "similar_case";
  readonly model_name: string;
  readonly model_version: string;
  readonly label_id: number | null;
  readonly label_name: string | null;
  readonly confidence: number | null;
  readonly uncertainty: number | null;
  readonly ood_score: number | null;
  readonly top_factors: readonly EvidenceFactor[];
  readonly explanation: string;
};

export type FusionSummaryPayload = {
  readonly strategy: string;
  readonly final_label_id: number | null;
  readonly final_label_name: string | null;
  readonly confidence: number | null;
  readonly agreement_level: string;
  readonly contributing_sources: readonly string[];
  readonly rationale: string;
  readonly evidence: readonly StandardModelEvidence[];
};

export type DiagnosisListItem = {
  readonly diagnosis_id: string;
  readonly trace_id: string;
  readonly route: InputRoute;
  readonly status: DiagnosisStatus;
  readonly diagnosis: string | null;
  readonly risk_level: string | null;
  readonly confidence: number | null;
  readonly reason: string;
  readonly requires_human_review: boolean;
  readonly created_at: string;
};

export type DiagnosisListResponse = {
  readonly items: readonly DiagnosisListItem[];
};

export type CaseTimelineEvent = {
  readonly kind: "diagnosis" | "trace";
  readonly title: string;
  readonly body: string;
  readonly created_at: string;
};

export type DiagnosisDetailResponse = {
  readonly diagnosis: DiagnosisListItem;
  readonly trace: TraceResponse;
  readonly timeline: readonly CaseTimelineEvent[];
};

export type DiagnosisReportResponse = {
  readonly detail: DiagnosisDetailResponse;
  readonly reference_cases: readonly SimilarCase[];
};

export type RagDocument = {
  readonly document_id: string;
  readonly title: string;
  readonly source: string;
  readonly excerpt: string;
  readonly relevance: number;
  readonly source_type: string | null;
  readonly retrieval_mode: string | null;
  readonly metadata: Record<string, string | number | null>;
};

export type RagSourceCount = {
  readonly documents: number;
  readonly chunks: number;
};

export type RagAppliedFilter = {
  readonly key: string;
  readonly label: string;
  readonly value: string;
};

export type RagStatusResponse = {
  readonly ready: boolean;
  readonly database_connected: boolean;
  readonly database_name: string;
  readonly vector_extension: string | null;
  readonly embedding_model: string;
  readonly vector_dim: number;
  readonly top_k: number;
  readonly source_types: readonly string[];
  readonly document_count: number;
  readonly chunk_count: number;
  readonly query_log_count: number;
  readonly source_counts: Record<string, RagSourceCount>;
  readonly last_indexed_at: string | null;
  readonly metadata_missing_counts: Record<string, number>;
  readonly error: string | null;
};

export type RagDocumentListItem = {
  readonly document_key: string;
  readonly source_type: string;
  readonly title: string;
  readonly source_path: string | null;
  readonly updated_at: string;
  readonly chunk_count: number;
};

export type RagDocumentChunk = {
  readonly chunk_key: string;
  readonly chunk_index: number;
  readonly text: string;
  readonly source_ref: string | null;
  readonly metadata: Record<string, string | number | null>;
};

export type RagDocumentDetailResponse = {
  readonly document_key: string;
  readonly source_type: string;
  readonly title: string;
  readonly source_path: string | null;
  readonly updated_at: string;
  readonly metadata: Record<string, string | number | null>;
  readonly chunks: readonly RagDocumentChunk[];
  readonly text: string;
  readonly error: string | null;
};

export type RagDocumentListResponse = {
  readonly items: readonly RagDocumentListItem[];
  readonly error: string | null;
};

export type RagQueryLogItem = {
  readonly id: number;
  readonly diagnosis_id: string | null;
  readonly query_text: string;
  readonly query_metadata: Record<string, unknown>;
  readonly retrieved_chunks: readonly unknown[];
  readonly created_at: string;
};

export type RagQueryLogResponse = {
  readonly items: readonly RagQueryLogItem[];
  readonly error: string | null;
};

export type RagSearchResponse = {
  readonly query: string;
  readonly documents: readonly RagDocument[];
  readonly applied_filters: readonly RagAppliedFilter[];
  readonly retrieval_mode: string | null;
  readonly result_count: number;
  readonly error: string | null;
};

export type RagChatMessage = {
  readonly role: "user" | "assistant";
  readonly content: string;
};

export type RagChatResponse = {
  readonly answer: string;
  readonly answer_mode: string | null;
  readonly documents: readonly RagDocument[];
  readonly model: string | null;
  readonly ready: boolean;
  readonly error: string | null;
};

export type RagReindexResponse = {
  readonly document_count: number;
  readonly chunk_count: number;
  readonly dataset_limit: number | null;
  readonly embedding_model: string;
};

export type SimilarCase = {
  readonly sample_id: string;
  readonly label_id: number;
  readonly label_name: string;
  readonly equipment_name: string;
  readonly insulator_type: string;
  readonly sensor_type: string;
  readonly clearance_distance: string;
  readonly similarity: number;
  readonly reason: string;
  readonly image_url: string;
  readonly timeseries_url: string | null;
  readonly metadata: Record<string, string | number | null>;
};

export type HealthResponse = {
  readonly status: string;
  readonly agent_mode: string;
  readonly agents_sdk_installed: boolean;
  readonly agents_sdk_reason: string;
};

export type ModelRuntimeStatus = {
  readonly agent_mode: string;
  readonly agents_sdk_installed: boolean;
  readonly agents_sdk_reason: string;
  readonly adapter_mode: string;
  readonly artifact_root: string;
  readonly time_series_model: string;
  readonly time_series_version: string;
  readonly time_series_adapter: string;
  readonly time_series_ready: boolean;
  readonly time_series_manifest: string | null;
  readonly time_series_checkpoint: string | null;
  readonly time_series_error: string | null;
  readonly vision_model: string;
  readonly vision_version: string;
  readonly vision_adapter: string;
  readonly vision_ready: boolean;
  readonly vision_manifest: string | null;
  readonly vision_checkpoint: string | null;
  readonly vision_error: string | null;
  readonly case_retriever: string;
  readonly case_version: string;
  readonly rag_retriever: string;
  readonly rag_version: string;
  readonly vlm_model: string;
  readonly vlm_version: string;
  readonly vlm_adapter: string;
  readonly vlm_ready: boolean;
  readonly vlm_manifest: string | null;
  readonly vlm_checkpoint: string | null;
  readonly vlm_error: string | null;
  readonly llm_rag_provider: string;
  readonly llm_rag_adapter: string;
  readonly llm_rag_ready: boolean;
  readonly llm_rag_model: string | null;
  readonly llm_rag_error: string | null;
};
