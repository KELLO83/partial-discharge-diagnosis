import type {
  DiagnosisListResponse,
  DiagnosisResponse,
  HealthResponse,
  MetadataForm,
  ModelRuntimeStatus,
  TraceResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

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

export async function fetchModelStatus(): Promise<ModelRuntimeStatus> {
  const response = await fetch(`${API_BASE}/model-status`);
  if (!response.ok) {
    throw new Error(`model status request failed: ${response.status}`);
  }
  return await response.json() as ModelRuntimeStatus;
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
  return {
    equipment_name: metadata.equipmentName,
    equipment_rated_voltage: metadata.ratedVoltage,
    equipment_rated_current: metadata.ratedCurrent,
    sensor_type: metadata.sensorType,
    temperature: Number(metadata.temperature),
    humidity: Number(metadata.humidity),
  };
}
