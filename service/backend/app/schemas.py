from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DiagnosisRoute = Literal["insufficient_input", "timeseries_only", "vlm_only", "hybrid"]
DiagnosisStatus = Literal["completed", "needs_review", "rejected"]


class MetadataInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    equipment_name: str = Field(min_length=1)
    equipment_rated_voltage: str = Field(min_length=1)
    equipment_rated_current: str = Field(min_length=1)
    sensor_type: str = Field(min_length=1)
    temperature: float
    humidity: float
    insulator_type: str | None = None
    insulator_name: str | None = None
    clearance_distance: str | None = None


class TimeSeriesResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    label_id: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    features: dict[str, float]


class VlmResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    label_id: int
    diagnosis: str
    risk_level: str
    confidence: float
    reason: str
    recommended_action: str


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


class TraceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    diagnosis_id: str
    trace_id: str
    route: DiagnosisRoute
    status: DiagnosisStatus
    steps: list[str]
    summary: dict[str, str]
