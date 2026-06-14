from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path

from prpd_similarity_retrieval import FEATURE_SCHEMA_VERSION
from prpd_similarity_retrieval.compact_index import CompactFeatureIndex, is_compact_index_path, load_compact_feature_index
from prpd_similarity_retrieval.features import extract_case_features
from prpd_similarity_retrieval.models import CaseRecord, SearchResult
from prpd_similarity_retrieval.retrieval import load_feature_index, search_similar_cases
from service.backend.app.domain.policy import label_name
from service.backend.app.schemas import MetadataInput, SimilarCase, SimilarCaseResult, TimeSeriesResult, VisionResult
from service.backend.app.domain.similar_cases import build_similarity_query, dataset_case_repository
from service.backend.app.application.contracts import SimilarCaseRetrievalAdapter, SimilarCaseToolInput


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPACT_INDEX_PATH = PROJECT_ROOT / "prpd_similarity_retrieval" / "case_feature_index.npz"
DEFAULT_JSON_INDEX_PATH = PROJECT_ROOT / "prpd_similarity_retrieval" / "case_feature_index.json"
SAMPLE_COMPACT_INDEX_PATH = PROJECT_ROOT / "prpd_similarity_retrieval" / "case_feature_index.sample.npz"
SAMPLE_JSON_INDEX_PATH = PROJECT_ROOT / "prpd_similarity_retrieval" / "case_feature_index.sample.json"
DEFAULT_CASE_LIMIT = 5


class FeatureSimilarityCaseRetrievalAdapter(SimilarCaseRetrievalAdapter):
    model_name = "domain_feature_case_retriever"
    model_version = FEATURE_SCHEMA_VERSION

    @cached_property
    def _indexed_cases(self):
        index_path = _feature_index_path()
        if index_path is None:
            return None
        return _load_index(index_path)

    def run(self, tool_input: SimilarCaseToolInput) -> SimilarCaseResult:
        indexed_cases = self._indexed_cases
        if indexed_cases is None:
            return _metadata_fallback_result(tool_input)
        query = _query_features(tool_input)
        results = _search_index(indexed_cases, query)
        return SimilarCaseResult(
            retriever_name=self.model_name,
            retriever_version=self.model_version,
            query=_feature_query_text(tool_input),
            cases=[_to_backend_case(result) for result in results],
        )


def _feature_index_path() -> Path | None:
    raw_path = os.getenv("PRPD_CASE_FEATURE_INDEX")
    candidates = (
        [Path(raw_path)]
        if raw_path
        else [DEFAULT_COMPACT_INDEX_PATH, DEFAULT_JSON_INDEX_PATH, SAMPLE_COMPACT_INDEX_PATH, SAMPLE_JSON_INDEX_PATH]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_index(path: Path):
    if is_compact_index_path(path):
        return load_compact_feature_index(path)
    return load_feature_index(path)


def _search_index(index, query):
    if isinstance(index, CompactFeatureIndex):
        return index.search_similar_cases(query, top_k=DEFAULT_CASE_LIMIT, exclude_self=False)
    return search_similar_cases(query, index, top_k=DEFAULT_CASE_LIMIT, exclude_self=False)


def _query_features(tool_input: SimilarCaseToolInput):
    label_id = _candidate_label_id(tool_input.timeseries_result, tool_input.vision_result)
    return extract_case_features(
        CaseRecord(
            sample_id="current_inspection",
            label_id=label_id,
            label_name=label_name(label_id) if label_id is not None else "",
            image_path=tool_input.image_path,
            timeseries_path=tool_input.timeseries_path,
            metadata=_metadata_dict(tool_input.safe_metadata),
        )
    )


def _candidate_label_id(
    timeseries_result: TimeSeriesResult | None,
    vision_result: VisionResult | None,
) -> int | None:
    if timeseries_result is not None and vision_result is not None and timeseries_result.label_id == vision_result.label_id:
        return timeseries_result.label_id
    if timeseries_result is not None:
        return timeseries_result.label_id
    if vision_result is not None:
        return vision_result.label_id
    return None


def _metadata_dict(metadata: MetadataInput | None) -> dict[str, str]:
    if metadata is None:
        return {}
    return {
        "equipment_name": metadata.equipment_name,
        "equipment_type": metadata.equipment_type or "",
        "equipment_rated_voltage": metadata.equipment_rated_voltage,
        "equipment_rated_current": metadata.equipment_rated_current,
        "insulator_type": metadata.insulator_type or "",
        "insulator_name": metadata.insulator_name or "",
        "sensor_type": metadata.sensor_type,
        "clearance_distance": metadata.clearance_distance or "",
    }


def _to_backend_case(result: SearchResult) -> SimilarCase:
    metadata = result.case.metadata
    return SimilarCase(
        sample_id=result.case.sample_id,
        label_id=result.case.label_id or -1,
        label_name=result.case.label_name,
        equipment_name=metadata.get("equipment_name", ""),
        insulator_type=metadata.get("insulator_type", ""),
        sensor_type=metadata.get("sensor_type", ""),
        clearance_distance=metadata.get("clearance_distance", ""),
        similarity=round(result.score, 6),
        reason=result.reason,
        image_url=f"/dataset/cases/{result.case.sample_id}/image",
        metadata={
            "equipment_rated_voltage": metadata.get("equipment_rated_voltage", ""),
            "equipment_rated_current": metadata.get("equipment_rated_current", ""),
            "feature_component_prpd": result.image_score,
            "feature_component_timeseries": result.timeseries_score,
            "feature_component_metadata": result.metadata_score,
            "feature_component_label": result.label_score,
        },
    )


def _feature_query_text(tool_input: SimilarCaseToolInput) -> str:
    return (
        f"route={tool_input.route}; "
        f"image={tool_input.image_path is not None}; "
        f"timeseries={tool_input.timeseries_path is not None}; "
        "feature=prpd_image+timeseries+metadata+label"
    )


def _metadata_fallback_result(tool_input: SimilarCaseToolInput) -> SimilarCaseResult:
    cases = dataset_case_repository.similar_cases(
        tool_input.safe_metadata,
        tool_input.timeseries_result,
        tool_input.vision_result,
    )
    return SimilarCaseResult(
        retriever_name="metadata_weighted_case_retriever_fallback",
        retriever_version="legacy",
        query=build_similarity_query(
            tool_input.safe_metadata,
            tool_input.timeseries_result,
            tool_input.vision_result,
        ),
        cases=cases,
    )
