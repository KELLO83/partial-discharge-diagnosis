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
