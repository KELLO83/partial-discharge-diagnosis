import type { DiagnosisResponse, MetadataForm } from "./types";

const API_BASE = "http://127.0.0.1:8000";

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
