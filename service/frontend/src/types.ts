import type { InputRoute } from "./route";

export type MetadataForm = {
  readonly equipmentName: string;
  readonly ratedVoltage: string;
  readonly ratedCurrent: string;
  readonly sensorType: string;
  readonly temperature: string;
  readonly humidity: string;
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
  readonly time_series_model: string;
  readonly time_series_version: string;
  readonly vlm_model: string;
  readonly vlm_version: string;
};
