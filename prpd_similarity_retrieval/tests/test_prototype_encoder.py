from __future__ import annotations

from pathlib import Path

from prpd_similarity_retrieval.compact_index import load_compact_feature_index, save_compact_feature_index
from prpd_similarity_retrieval.models import CaseFeatures
from prpd_similarity_retrieval.prototype_encoder import (
    PrototypeEncoderConfig,
    build_prototype_embedding_index,
    evaluate_prototype_index,
    load_prototype_embedding_index,
    save_prototype_embedding_index,
)


def test_prototype_index_round_trips_case_count(tmp_path: Path) -> None:
    prototype_index = _prototype_index(tmp_path, _cases())
    output_path = tmp_path / "case_embedding_index.prototype.npz"
    save_prototype_embedding_index(output_path, prototype_index)

    loaded = load_prototype_embedding_index(output_path)

    assert loaded.case_count == 4


def test_prototype_search_returns_same_label_neighbor(tmp_path: Path) -> None:
    prototype_index = _prototype_index(tmp_path, _cases())

    result = prototype_index.search_sample("query", top_k=1)[0]

    assert result.case.label_id == 1


def test_prototype_evaluation_reports_label_match(tmp_path: Path) -> None:
    prototype_index = _prototype_index(tmp_path, _cases())

    metrics = evaluate_prototype_index(prototype_index, limit=None, top_k=1)

    assert metrics.top1_label_match_rate == 1.0


def test_prototype_evaluation_skips_singleton_index(tmp_path: Path) -> None:
    prototype_index = _prototype_index(tmp_path, [_case("only", 1, [1.0, 0.0])])

    metrics = evaluate_prototype_index(prototype_index, limit=None, top_k=1)

    assert metrics.evaluated == 0


def _prototype_index(tmp_path: Path, cases: list[CaseFeatures]):
    feature_path = tmp_path / "case_feature_index.npz"
    save_compact_feature_index(feature_path, cases)
    feature_index = load_compact_feature_index(feature_path)
    return build_prototype_embedding_index(
        feature_index,
        PrototypeEncoderConfig(image_dim=8, timeseries_dim=4, centroid_weight=0.25, random_seed=7),
    )


def _cases() -> list[CaseFeatures]:
    return [
        _case("query", 1, [1.0, 0.0]),
        _case("near", 1, [0.95, 0.05]),
        _case("far", 2, [0.0, 1.0]),
        _case("far-near", 2, [0.05, 0.95]),
    ]


def _case(sample_id: str, label_id: int, vector: list[float]) -> CaseFeatures:
    return CaseFeatures(
        sample_id=sample_id,
        label_id=label_id,
        label_name=f"label-{label_id}",
        image_path=f"{sample_id}.png",
        timeseries_path=f"{sample_id}.csv",
        metadata={"sensor_type": "HFCT"},
        image_vector=vector,
        timeseries_vector=vector,
    )
