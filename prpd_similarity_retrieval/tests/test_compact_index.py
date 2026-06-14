from __future__ import annotations

from pathlib import Path

import pytest

from prpd_similarity_retrieval.compact_index import load_compact_feature_index, save_compact_feature_index
from prpd_similarity_retrieval.models import CaseFeatures
from prpd_similarity_retrieval.retrieval import search_similar_cases


def test_compact_search_preserves_json_candidate_order(tmp_path: Path) -> None:
    cases = _cases()
    index_path = tmp_path / "case_feature_index.npz"
    save_compact_feature_index(index_path, cases)
    index = load_compact_feature_index(index_path)

    query = index.find_case("query")
    expected = search_similar_cases(query, cases, top_k=2)
    actual = index.search_similar_cases(query, top_k=2)

    assert [result.case.sample_id for result in actual] == [result.case.sample_id for result in expected]


def test_compact_search_preserves_json_scores(tmp_path: Path) -> None:
    cases = _cases()
    index_path = tmp_path / "case_feature_index.npz"
    save_compact_feature_index(index_path, cases)
    index = load_compact_feature_index(index_path)

    query = index.find_case("query")
    expected = search_similar_cases(query, cases, top_k=2)
    actual = index.search_similar_cases(query, top_k=2)

    assert [result.score for result in actual] == pytest.approx([result.score for result in expected])


def test_compact_index_restores_query_vectors(tmp_path: Path) -> None:
    index_path = tmp_path / "case_feature_index.npz"
    save_compact_feature_index(index_path, _cases())
    index = load_compact_feature_index(index_path)

    assert index.find_case("query").image_vector == [1.0, 0.0]


def test_compact_metadata_baseline_omits_image_score(tmp_path: Path) -> None:
    index_path = tmp_path / "case_feature_index.npz"
    save_compact_feature_index(index_path, _cases())
    index = load_compact_feature_index(index_path)

    result = index.search_metadata_baseline(index.find_case("query"), top_k=1)[0]

    assert result.image_score is None


def test_compact_search_matches_json_score_when_metadata_is_missing(tmp_path: Path) -> None:
    cases = [
        _case("query", 1, [1.0, 0.0], [1.0, 0.0], "1000mm"),
        _case("missing-metadata", 1, [1.0, 0.0], [1.0, 0.0], ""),
    ]
    index_path = tmp_path / "case_feature_index.npz"
    save_compact_feature_index(index_path, cases)
    index = load_compact_feature_index(index_path)

    expected = search_similar_cases(cases[0], cases, top_k=1)[0]
    actual = index.search_similar_cases(index.find_case("query"), top_k=1)[0]

    assert actual.score == pytest.approx(expected.score)


def test_compact_search_preserves_json_tie_order(tmp_path: Path) -> None:
    cases = [
        _case("query", 1, [1.0, 0.0], [1.0, 0.0], "1000mm"),
        _case("aaa", 1, [1.0, 0.0], [1.0, 0.0], "1000mm"),
        _case("zzz", 1, [1.0, 0.0], [1.0, 0.0], "1000mm"),
    ]
    index_path = tmp_path / "case_feature_index.npz"
    save_compact_feature_index(index_path, cases)
    index = load_compact_feature_index(index_path)

    expected = search_similar_cases(cases[0], cases, top_k=2)
    actual = index.search_similar_cases(index.find_case("query"), top_k=2)

    assert [result.case.sample_id for result in actual] == [result.case.sample_id for result in expected]


def _cases() -> list[CaseFeatures]:
    return [
        _case("query", 1, [1.0, 0.0], [1.0, 0.0], "1000mm"),
        _case("near", 1, [0.95, 0.05], [0.90, 0.10], "1000mm"),
        _case("far", 2, [0.0, 1.0], [0.0, 1.0], "500mm"),
    ]


def _case(
    sample_id: str,
    label_id: int,
    image_vector: list[float],
    timeseries_vector: list[float],
    clearance_distance: str,
) -> CaseFeatures:
    return CaseFeatures(
        sample_id=sample_id,
        label_id=label_id,
        label_name=f"label-{label_id}",
        image_path=f"{sample_id}.png",
        timeseries_path=f"{sample_id}.csv",
        metadata={
            "equipment_name": "ACSR-OC",
            "sensor_type": "HFCT",
            "insulator_type": "solid",
            "clearance_distance": clearance_distance,
            "equipment_rated_voltage": "22900V",
            "equipment_rated_current": "268A",
        },
        image_vector=image_vector,
        timeseries_vector=timeseries_vector,
    )
