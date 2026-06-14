from __future__ import annotations

from pathlib import Path

from prpd_similarity_retrieval.feature_cache import append_feature_cache, load_feature_cache, load_feature_cache_map
from prpd_similarity_retrieval.models import CaseFeatures


def test_empty_feature_cache_returns_empty_list(tmp_path: Path) -> None:
    assert load_feature_cache(tmp_path / "missing.jsonl") == []


def test_feature_cache_round_trips_sample_id(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    append_feature_cache(cache_path, [_case("sample-a", [1.0, 0.0])])

    cached = load_feature_cache(cache_path)

    assert cached[0].sample_id == "sample-a"


def test_feature_cache_map_keeps_latest_duplicate(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    append_feature_cache(cache_path, [_case("sample-a", [1.0, 0.0])])
    append_feature_cache(cache_path, [_case("sample-a", [0.0, 1.0])])

    cached = load_feature_cache_map(cache_path)

    assert cached["sample-a"].image_vector == [0.0, 1.0]


def _case(sample_id: str, image_vector: list[float]) -> CaseFeatures:
    return CaseFeatures(
        sample_id=sample_id,
        label_id=1,
        label_name="noise",
        image_path=f"{sample_id}.png",
        timeseries_path=f"{sample_id}.csv",
        metadata={"sensor_type": "HFCT"},
        image_vector=image_vector,
        timeseries_vector=[1.0, 0.0],
    )
