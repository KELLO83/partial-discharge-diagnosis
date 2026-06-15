import type {
  DiagnosisDetailResponse,
  DiagnosisListResponse,
  DiagnosisReportResponse,
  DiagnosisResponse,
  HealthResponse,
  MetadataForm,
  ModelRuntimeStatus,
  RagChatMessage,
  RagChatResponse,
  RagDocumentDetailResponse,
  RagDocumentListResponse,
  RagQueryLogResponse,
  RagReindexResponse,
  RagSearchResponse,
  RagStatusResponse,
  SimilarCase,
  TraceResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8001";

export function apiAssetUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${API_BASE}${path}`;
}

export async function submitDiagnosis(input: {
  readonly image: File | null;
  readonly csv: File | null;
  readonly metadata: MetadataForm;
}): Promise<DiagnosisResponse> {
  const form = new FormData();
  if (input.image !== null) {
    form.append("prpd_image", input.image);
  }
  if (input.csv !== null) {
    form.append("timeseries_csv", input.csv);
  }
  if (metadataComplete(input.metadata)) {
    form.append("metadata", JSON.stringify(toApiMetadata(input.metadata)));
  }
  const response = await fetch(`${API_BASE}/diagnose`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw new Error(`diagnose request failed: ${response.status}`);
  }
  return await response.json() as DiagnosisResponse;
}

export async function fetchDiagnosisTrace(diagnosisId: string): Promise<TraceResponse> {
  const response = await fetch(`${API_BASE}/diagnose/${diagnosisId}/trace`);
  if (!response.ok) {
    throw new Error(`trace request failed: ${response.status}`);
  }
  return await response.json() as TraceResponse;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error(`health request failed: ${response.status}`);
  }
  return await response.json() as HealthResponse;
}

export async function fetchModelRuntimeStatus(): Promise<ModelRuntimeStatus> {
  const response = await fetch(`${API_BASE}/model-status`);
  if (!response.ok) {
    throw new Error(`model status request failed: ${response.status}`);
  }
  return await response.json() as ModelRuntimeStatus;
}

export async function fetchRagStatus(): Promise<RagStatusResponse> {
  const response = await fetch(`${API_BASE}/rag/status`);
  if (!response.ok) {
    throw new Error(`rag status request failed: ${response.status}`);
  }
  return await response.json() as RagStatusResponse;
}

export async function fetchRagDocuments(input: {
  readonly sourceType: string;
  readonly limit?: number;
}): Promise<RagDocumentListResponse> {
  const params = new URLSearchParams({limit: String(input.limit ?? 50)});
  if (input.sourceType !== "all") {
    params.set("source_type", input.sourceType);
  }
  const response = await fetch(`${API_BASE}/rag/documents?${params.toString()}`);
  if (!response.ok) {
    throw new Error(`rag documents request failed: ${response.status}`);
  }
  return await response.json() as RagDocumentListResponse;
}

export async function fetchRagDocumentDetail(documentKey: string): Promise<RagDocumentDetailResponse> {
  const response = await fetch(`${API_BASE}/rag/documents/${encodeURIComponent(documentKey)}`);
  if (!response.ok) {
    throw new Error(`rag document detail request failed: ${response.status}`);
  }
  return await response.json() as RagDocumentDetailResponse;
}

export async function fetchRagQueryLogs(limit = 20): Promise<RagQueryLogResponse> {
  const response = await fetch(`${API_BASE}/rag/query-logs?limit=${limit}`);
  if (!response.ok) {
    throw new Error(`rag query logs request failed: ${response.status}`);
  }
  return await response.json() as RagQueryLogResponse;
}

export async function searchRagDocuments(input: {
  readonly query: string;
  readonly topK: number;
}): Promise<RagSearchResponse> {
  const response = await fetch(`${API_BASE}/rag/search`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({query: input.query, top_k: input.topK}),
  });
  if (!response.ok) {
    throw new Error(`rag search request failed: ${response.status}`);
  }
  return await response.json() as RagSearchResponse;
}

export async function askRagChat(input: {
  readonly messages: readonly RagChatMessage[];
  readonly question: string;
  readonly topK: number;
}): Promise<RagChatResponse> {
  const response = await fetch(`${API_BASE}/rag/chat`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      messages: input.messages,
      question: input.question,
      top_k: input.topK,
    }),
  });
  if (!response.ok) {
    throw new Error(`rag chat request failed: ${response.status}`);
  }
  return await response.json() as RagChatResponse;
}

export async function reindexRagDocuments(datasetLimit: number | null): Promise<RagReindexResponse> {
  const response = await fetch(`${API_BASE}/rag/reindex`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({dataset_limit: datasetLimit}),
  });
  if (!response.ok) {
    throw new Error(`rag reindex request failed: ${response.status}`);
  }
  return await response.json() as RagReindexResponse;
}

export async function fetchDiagnosisHistory(): Promise<DiagnosisListResponse> {
  const response = await fetch(`${API_BASE}/diagnoses`);
  if (!response.ok) {
    throw new Error(`diagnosis history request failed: ${response.status}`);
  }
  return await response.json() as DiagnosisListResponse;
}

export async function fetchReviewQueue(): Promise<DiagnosisListResponse> {
  const response = await fetch(`${API_BASE}/review-queue`);
  if (!response.ok) {
    throw new Error(`review queue request failed: ${response.status}`);
  }
  return await response.json() as DiagnosisListResponse;
}

export async function fetchDiagnosisDetail(diagnosisId: string): Promise<DiagnosisDetailResponse> {
  const response = await fetch(`${API_BASE}/diagnoses/${diagnosisId}`);
  if (!response.ok) {
    throw new Error(`diagnosis detail request failed: ${response.status}`);
  }
  return await response.json() as DiagnosisDetailResponse;
}

export async function fetchDiagnosisReport(diagnosisId: string): Promise<DiagnosisReportResponse> {
  const response = await fetch(`${API_BASE}/diagnoses/${diagnosisId}/report`);
  if (!response.ok) {
    throw new Error(`diagnosis report request failed: ${response.status}`);
  }
  return await response.json() as DiagnosisReportResponse;
}

export async function fetchDatasetCaseDetail(sampleId: string): Promise<SimilarCase> {
  const response = await fetch(`${API_BASE}/dataset/cases/${encodeURIComponent(sampleId)}`);
  if (!response.ok) {
    throw new Error(`dataset case detail request failed: ${response.status}`);
  }
  return await response.json() as SimilarCase;
}

function metadataComplete(metadata: MetadataForm): boolean {
  return [
    metadata.equipmentName,
    metadata.ratedVoltage,
    metadata.ratedCurrent,
    metadata.sensorType,
    metadata.temperature,
    metadata.humidity,
  ].every((value) => value.trim().length > 0);
}

function toApiMetadata(metadata: MetadataForm): Record<string, string | number> {
  const payload: Record<string, string | number> = {
    equipment_name: metadata.equipmentName,
    equipment_rated_voltage: metadata.ratedVoltage,
    equipment_rated_current: metadata.ratedCurrent,
    sensor_type: metadata.sensorType,
    temperature: Number(metadata.temperature),
    humidity: Number(metadata.humidity),
  };
  addOptionalString(payload, "equipment_type", metadata.equipmentType);
  addOptionalString(payload, "measurement_location", metadata.measurementLocation);
  addOptionalString(payload, "operating_condition", metadata.operatingCondition);
  addOptionalString(payload, "insulator_type", metadata.insulatorType);
  addOptionalString(payload, "clearance_distance", metadata.clearanceDistance);
  return payload;
}

function addOptionalString(payload: Record<string, string | number>, key: string, value: string | undefined): void {
  if (value !== undefined && value.trim().length > 0) {
    payload[key] = value;
  }
}
