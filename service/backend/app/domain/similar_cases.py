from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from urllib.parse import quote

from service.backend.app.schemas import MetadataInput, SimilarCase, TimeSeriesResult, VisionResult


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_ROOT / "manifest.csv"
DEFAULT_CASE_LIMIT = 5
LABEL_SCORE_WEIGHT = 0.38
EQUIPMENT_SCORE_WEIGHT = 0.18
INSULATOR_SCORE_WEIGHT = 0.10
SENSOR_SCORE_WEIGHT = 0.08
CLEARANCE_SCORE_WEIGHT = 0.08
VOLTAGE_SCORE_WEIGHT = 0.06
SIGNAL_SCORE_WEIGHT = 0.12


@dataclass(frozen=True, slots=True)
class DatasetCase:
    sample_id: str
    label_id: int
    label_name: str
    equipment_name: str
    insulator_type: str
    sensor_type: str
    clearance_distance: str
    equipment_rated_voltage: str
    equipment_rated_current: str
    temperature: str
    humidity: str
    max_discharge_value: float | None
    json_path: Path
    image_path: Path
    timeseries_path: Path


@dataclass(frozen=True, slots=True)
class QueryEvidence:
    metadata: MetadataInput | None
    label_ids: tuple[int, ...]
    abs_p99: float | None


class DatasetCaseRepository:
    def __init__(
        self,
        manifest_path: Path = DEFAULT_MANIFEST_PATH,
        data_root: Path = DEFAULT_DATA_ROOT,
    ) -> None:
        self.manifest_path = manifest_path
        self.data_root = data_root

    @cached_property
    def cases(self) -> tuple[DatasetCase, ...]:
        if not self.manifest_path.exists():
            return ()
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return tuple(self._row_to_case(row) for row in csv.DictReader(handle))

    def list(self, limit: int = 20) -> list[DatasetCase]:
        return list(self.cases[:limit])

    def get(self, sample_id: str) -> DatasetCase | None:
        return next((case for case in self.cases if case.sample_id == sample_id), None)

    def search(
        self,
        *,
        label_id: int | None = None,
        equipment_name: str | None = None,
        sensor_type: str | None = None,
        insulator_type: str | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> list[SimilarCase]:
        filtered = [
            case
            for case in self.cases
            if _matches_case(case, label_id, equipment_name, sensor_type, insulator_type, query)
        ]
        return [
            to_similar_case(case, 1.0, "검색 조건과 일치하는 데이터셋 사례")
            for case in filtered[:limit]
        ]

    def similar_cases(
        self,
        metadata: MetadataInput | None,
        ts_result: TimeSeriesResult | None,
        vision_result: VisionResult | None,
        limit: int = DEFAULT_CASE_LIMIT,
    ) -> list[SimilarCase]:
        query = QueryEvidence(
            metadata=metadata,
            label_ids=_candidate_label_ids(ts_result, vision_result),
            abs_p99=_abs_p99(ts_result),
        )
        candidates = _cases_matching_candidate_labels(self.cases, query.label_ids)
        scored = [
            (_case_score(case, query), case)
            for case in candidates
        ]
        scored.sort(key=lambda item: (item[0], item[1].sample_id), reverse=True)
        return [
            to_similar_case(case, score, _case_reason(case, query))
            for score, case in scored[:limit]
        ]

    def _row_to_case(self, row: dict[str, str]) -> DatasetCase:
        return DatasetCase(
            sample_id=row["sample_id"],
            label_id=int(row["label_id"]),
            label_name=row["label_name"],
            equipment_name=row.get("equipment_name", ""),
            insulator_type=row.get("insulator_type", ""),
            sensor_type=row.get("sensor_type", ""),
            clearance_distance=row.get("clearance_distance", ""),
            equipment_rated_voltage=row.get("equipment_rated_voltage", ""),
            equipment_rated_current=row.get("equipment_rated_current", ""),
            temperature=row.get("temperature", ""),
            humidity=row.get("humidity", ""),
            max_discharge_value=_optional_float(row.get("max_discharge_value")),
            json_path=_resolve_dataset_path(row.get("json_path", ""), self.data_root),
            image_path=_resolve_dataset_path(row.get("image_path", ""), self.data_root),
            timeseries_path=_resolve_dataset_path(row.get("timeseries_path", ""), self.data_root),
        )


def build_similarity_query(
    metadata: MetadataInput | None,
    ts_result: TimeSeriesResult | None,
    vision_result: VisionResult | None,
) -> str:
    label_ids = _candidate_label_ids(ts_result, vision_result)
    equipment = metadata.equipment_name if metadata is not None else "unknown_equipment"
    sensor = metadata.sensor_type if metadata is not None else "unknown_sensor"
    labels = ",".join(str(label_id) for label_id in label_ids) if label_ids else "unknown"
    return f"equipment={equipment}; sensor={sensor}; evidence_labels={labels}; feature=metadata+signal_proxy"


def _candidate_label_ids(ts_result: TimeSeriesResult | None, vision_result: VisionResult | None) -> tuple[int, ...]:
    label_ids: list[int] = []
    if ts_result is not None:
        label_ids.append(ts_result.label_id)
    if vision_result is not None:
        label_ids.append(vision_result.label_id)
    return tuple(dict.fromkeys(label_ids))


def _cases_matching_candidate_labels(
    cases: tuple[DatasetCase, ...],
    label_ids: tuple[int, ...],
) -> tuple[DatasetCase, ...]:
    if not label_ids:
        return cases
    return tuple(case for case in cases if case.label_id in label_ids)


def _abs_p99(ts_result: TimeSeriesResult | None) -> float | None:
    if ts_result is None:
        return None
    return ts_result.features.get("abs_p99")


def _case_score(case: DatasetCase, query: QueryEvidence) -> float:
    score = 0.0
    if case.label_id in query.label_ids:
        score += LABEL_SCORE_WEIGHT
    if query.metadata is not None:
        score += _metadata_score(case, query.metadata)
    if query.abs_p99 is not None and case.max_discharge_value is not None:
        score += SIGNAL_SCORE_WEIGHT * _numeric_similarity(query.abs_p99, case.max_discharge_value)
    return round(min(score, 0.99), 4)


def _metadata_score(case: DatasetCase, metadata: MetadataInput) -> float:
    score = 0.0
    if _same(case.equipment_name, metadata.equipment_name):
        score += EQUIPMENT_SCORE_WEIGHT
    if _same(case.insulator_type, metadata.insulator_type):
        score += INSULATOR_SCORE_WEIGHT
    if _same(case.sensor_type, metadata.sensor_type):
        score += SENSOR_SCORE_WEIGHT
    if _same(case.clearance_distance, metadata.clearance_distance):
        score += CLEARANCE_SCORE_WEIGHT
    if _same(case.equipment_rated_voltage, metadata.equipment_rated_voltage):
        score += VOLTAGE_SCORE_WEIGHT
    return score


def _case_reason(case: DatasetCase, query: QueryEvidence) -> str:
    reasons = []
    if case.label_id in query.label_ids:
        reasons.append("모델 근거 라벨 일치")
    if query.metadata is not None and _same(case.equipment_name, query.metadata.equipment_name):
        reasons.append("동일 설비")
    if query.metadata is not None and _same(case.sensor_type, query.metadata.sensor_type):
        reasons.append("동일 센서")
    if query.abs_p99 is not None and case.max_discharge_value is not None:
        reasons.append("신호 크기 지표 유사")
    return ", ".join(reasons) if reasons else "가용 메타데이터 기준 최인접 데이터셋 사례"


def to_similar_case(case: DatasetCase, score: float, reason: str) -> SimilarCase:
    return SimilarCase(
        sample_id=case.sample_id,
        label_id=case.label_id,
        label_name=case.label_name,
        equipment_name=case.equipment_name,
        insulator_type=case.insulator_type,
        sensor_type=case.sensor_type,
        clearance_distance=case.clearance_distance,
        similarity=score,
        reason=reason,
        image_url=f"/dataset/cases/{quote(case.sample_id, safe='')}/image",
        timeseries_url=f"/dataset/cases/{quote(case.sample_id, safe='')}/timeseries" if case.timeseries_path.exists() else None,
        metadata={
            "equipment_rated_voltage": case.equipment_rated_voltage,
            "equipment_rated_current": case.equipment_rated_current,
            "temperature": case.temperature,
            "humidity": case.humidity,
            "max_discharge_value": case.max_discharge_value,
        },
    )


def _matches_case(
    case: DatasetCase,
    label_id: int | None,
    equipment_name: str | None,
    sensor_type: str | None,
    insulator_type: str | None,
    query: str | None,
) -> bool:
    if label_id is not None and case.label_id != label_id:
        return False
    if equipment_name and _normalize(equipment_name) not in _normalize(case.equipment_name):
        return False
    if sensor_type and _normalize(sensor_type) != _normalize(case.sensor_type):
        return False
    if insulator_type and _normalize(insulator_type) != _normalize(case.insulator_type):
        return False
    if query and _normalize(query) not in _normalize(_case_search_text(case)):
        return False
    return True


def _case_search_text(case: DatasetCase) -> str:
    return " ".join(
        [
            case.sample_id,
            case.label_name,
            case.equipment_name,
            case.insulator_type,
            case.sensor_type,
            case.clearance_distance,
            case.equipment_rated_voltage,
        ]
    )


def _resolve_dataset_path(raw_path: str, data_root: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    parts = path.parts
    if parts and parts[0].lower() == "train":
        return data_root.joinpath(*parts[1:])
    return PROJECT_ROOT / path


def _numeric_similarity(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), 1.0)
    return max(0.0, 1.0 - min(abs(left - right) / denominator, 1.0))


def _same(left: str | None, right: str | None) -> bool:
    return _normalize(left) == _normalize(right) and _normalize(left) != ""


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    return value.lower().replace(" ", "").replace("['", "").replace("']", "").replace("mm", "")


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


dataset_case_repository = DatasetCaseRepository()
