from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PD_LABELS_KO: dict[int, str] = {
    0: "정상",
    1: "노이즈",
    2: "표면방전",
    3: "코로나방전",
    4: "보이드방전",
}

FORBIDDEN_PROMPT_FIELDS: tuple[str, ...] = (
    "label_id",
    "label_name",
    "PD_type",
    "sample_id",
    "image_path",
    "timeseries_path",
    "json_path",
    "defect_details",
    "defect_nums",
    "max_discharge_value",
)

SAFE_METADATA_FIELDS: tuple[str, ...] = (
    "equipment_name",
    "equipment_rated_voltage",
    "equipment_rated_current",
    "insulator_type",
    "insulator_name",
    "sensor_type",
    "temperature",
    "humidity",
    "clearance_distance",
)


@dataclass(frozen=True, slots=True)
class ManifestVlmRow:
    sample_id: str
    image_path: str
    label_id: int
    label_name: str
    equipment_name: str
    equipment_rated_voltage: str
    equipment_rated_current: str
    insulator_type: str
    insulator_name: str
    sensor_type: str
    temperature: str
    humidity: str
    clearance_distance: str
    defect_details: str
    defect_nums: str
    max_discharge_value: str
    split: str = "train"

    @classmethod
    def from_mapping(cls, row: dict[str, str]) -> ManifestVlmRow:
        return cls(
            sample_id=row.get("sample_id", ""),
            image_path=row.get("image_path", ""),
            label_id=int(row.get("label_id", "0")),
            label_name=row.get("label_name", ""),
            equipment_name=row.get("equipment_name", ""),
            equipment_rated_voltage=row.get("equipment_rated_voltage", ""),
            equipment_rated_current=row.get("equipment_rated_current", ""),
            insulator_type=row.get("insulator_type", ""),
            insulator_name=row.get("insulator_name", ""),
            sensor_type=row.get("sensor_type", ""),
            temperature=row.get("temperature", ""),
            humidity=row.get("humidity", ""),
            clearance_distance=row.get("clearance_distance", ""),
            defect_details=row.get("defect_details", ""),
            defect_nums=row.get("defect_nums", ""),
            max_discharge_value=row.get("max_discharge_value", ""),
            split=row.get("split", "train") or "train",
        )


@dataclass(frozen=True, slots=True)
class TimeSeriesContext:
    sample_id: str
    ts_model_name: str
    ts_pred_label_id: int | None
    ts_confidence: float | None
    ts_prob_0: float | None
    ts_prob_1: float | None
    ts_prob_2: float | None
    ts_prob_3: float | None
    ts_prob_4: float | None
    rms: float | None
    std: float | None
    abs_p99: float | None
    pulse_rate: float | None
    spectral_energy: float | None

    @classmethod
    def unavailable(cls, sample_id: str) -> TimeSeriesContext:
        return cls(
            sample_id=sample_id,
            ts_model_name="unavailable",
            ts_pred_label_id=None,
            ts_confidence=None,
            ts_prob_0=None,
            ts_prob_1=None,
            ts_prob_2=None,
            ts_prob_3=None,
            ts_prob_4=None,
            rms=None,
            std=None,
            abs_p99=None,
            pulse_rate=None,
            spectral_energy=None,
        )

    @classmethod
    def from_mapping(cls, row: dict[str, str]) -> TimeSeriesContext:
        return cls(
            sample_id=row.get("sample_id", ""),
            ts_model_name=row.get("ts_model_name", "feature_summary"),
            ts_pred_label_id=_optional_int(row.get("ts_pred_label_id", "")),
            ts_confidence=_optional_float(row.get("ts_confidence", "")),
            ts_prob_0=_optional_float(row.get("ts_prob_0", "")),
            ts_prob_1=_optional_float(row.get("ts_prob_1", "")),
            ts_prob_2=_optional_float(row.get("ts_prob_2", "")),
            ts_prob_3=_optional_float(row.get("ts_prob_3", "")),
            ts_prob_4=_optional_float(row.get("ts_prob_4", "")),
            rms=_optional_float(row.get("rms", "")),
            std=_optional_float(row.get("std", "")),
            abs_p99=_optional_float(row.get("abs_p99", "")),
            pulse_rate=_optional_float(row.get("pulse_rate", "")),
            spectral_energy=_optional_float(row.get("spectral_energy", "")),
        )


@dataclass(frozen=True, slots=True)
class DatasetBuildSummary:
    rows_written: int
    output_path: Path


@dataclass(frozen=True, slots=True)
class ValidationReport:
    valid: bool
    n_rows: int
    missing_images: int
    invalid_targets: int
    leakage_hits: int


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    n_rows: int
    json_parse_success_rate: float
    schema_validity_rate: float
    label_accuracy: float
    macro_f1: float
    parse_failures: int
    confusion_matrix: tuple[tuple[int, ...], ...]
    hallucinated_field_count: int
    forbidden_field_hit_count: int


def _optional_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def _optional_int(value: str) -> int | None:
    if value == "":
        return None
    return int(value)
