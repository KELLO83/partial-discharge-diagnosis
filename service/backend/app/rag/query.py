from __future__ import annotations

from dataclasses import dataclass

from service.backend.app.domain.policy import label_name
from service.backend.app.schemas import MetadataInput, SimilarCaseResult, TimeSeriesResult, VisionResult


@dataclass(frozen=True, slots=True)
class RagQueryInput:
    metadata: MetadataInput | None
    time_series: TimeSeriesResult | None
    vision: VisionResult | None
    similar_case: SimilarCaseResult | None


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    source: str
    label_id: int
    label: str
    confidence: float


def build_rag_query(input_data: RagQueryInput) -> str:
    fields = [
        _metadata_text(input_data.metadata),
        _time_series_text(input_data.time_series),
        _vision_text(input_data.vision),
        _vision_feature_text(input_data.vision),
        _similar_case_text(input_data.similar_case),
    ]
    return "\n".join(field for field in fields if field)


def candidate_label_ids(timeseries_result: TimeSeriesResult | None, vision_result: VisionResult | None) -> tuple[int, ...]:
    label_ids: list[int] = []
    if timeseries_result is not None:
        label_ids.append(timeseries_result.label_id)
    if vision_result is not None:
        label_ids.append(vision_result.label_id)
    return tuple(dict.fromkeys(label_ids))


def _metadata_text(metadata: MetadataInput | None) -> str:
    if metadata is None:
        return "metadata=unknown"
    return (
        f"equipment={metadata.equipment_name}; equipment_type={metadata.equipment_type or 'unknown'}; "
        f"sensor={metadata.sensor_type}; voltage={metadata.equipment_rated_voltage}; "
        f"insulator={metadata.insulator_type or metadata.insulator_name or 'unknown'}; "
        f"clearance={metadata.clearance_distance or 'unknown'}; "
        f"temperature={metadata.temperature}; humidity={metadata.humidity}"
    )


def _time_series_text(result: TimeSeriesResult | None) -> str:
    if result is None:
        return ""
    return _model_text(ModelCandidate("시계열", result.label_id, result.label_name, result.confidence))


def _vision_text(result: VisionResult | None) -> str:
    if result is None:
        return ""
    return _model_text(ModelCandidate("비전", result.label_id, result.label_name, result.confidence))


def _model_text(candidate: ModelCandidate) -> str:
    return (
        f"{candidate.source} 후보={candidate.label}({candidate.label_id}); "
        f"신뢰도={candidate.confidence:.2f}; 기준={label_name(candidate.label_id)}"
    )


def _vision_feature_text(result: VisionResult | None) -> str:
    if result is None:
        return ""
    items = "; ".join(f"{key}={value}" for key, value in result.evidence.items())
    return f"PRPD visual features: {items}"


def _similar_case_text(result: SimilarCaseResult | None) -> str:
    if result is None or not result.cases:
        return ""
    top_cases = ", ".join(
        f"{case.sample_id}:{case.label_name}:{case.similarity:.2f}"
        for case in result.cases[:3]
    )
    return f"similar historical cases: {top_cases}"
