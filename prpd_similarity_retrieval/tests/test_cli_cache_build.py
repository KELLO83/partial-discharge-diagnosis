from __future__ import annotations

from pathlib import Path

from prpd_similarity_retrieval.cli import _build_features
from prpd_similarity_retrieval.models import CaseRecord


def test_cached_build_reports_second_run_cache_hit(tmp_path: Path) -> None:
    cache_path = tmp_path / "feature_cache.jsonl"
    cases = [_case("sample-a")]
    _build_features(cases, progress_every=0, workers=1, cache_path=cache_path)

    result = _build_features(cases, progress_every=0, workers=1, cache_path=cache_path)

    assert result.cache_hit_count == 1


def test_cached_build_preserves_requested_order(tmp_path: Path) -> None:
    cache_path = tmp_path / "feature_cache.jsonl"
    cases = [_case("sample-b"), _case("sample-a")]

    result = _build_features(cases, progress_every=0, workers=1, cache_path=cache_path)

    assert [feature.sample_id for feature in result.features] == ["sample-b", "sample-a"]


def _case(sample_id: str) -> CaseRecord:
    return CaseRecord(
        sample_id=sample_id,
        label_id=1,
        label_name="noise",
        image_path=None,
        timeseries_path=None,
        metadata={"sensor_type": "HFCT"},
    )
