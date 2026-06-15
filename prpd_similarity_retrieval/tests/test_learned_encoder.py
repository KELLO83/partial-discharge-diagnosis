from __future__ import annotations

from pathlib import Path

from prpd_similarity_retrieval.compact_index import load_compact_feature_index, save_compact_feature_index
from prpd_similarity_retrieval.learned_encoder import (
    LearnedEncoderConfig,
    build_learned_embedding_index,
    evaluate_learned_index,
    load_learned_embedding_index,
    save_learned_embedding_index,
)
from prpd_similarity_retrieval.models import CaseFeatures


def test_learned_index_round_trips_case_count(tmp_path: Path) -> None:
    learned_index = _learned_index(tmp_path, _cases())
    output_path = tmp_path / "case_embedding_index.learned.npz"
    save_learned_embedding_index(output_path, learned_index)

    loaded = load_learned_embedding_index(output_path)

    assert loaded.case_count == 4
    assert loaded.embeddings.shape[0] == 4


def test_learned_search_returns_same_label_neighbor(tmp_path: Path) -> None:
    learned_index = _learned_index(tmp_path, _cases())

    result = learned_index.search_sample("query", top_k=1)[0]

    assert result.case.label_id == 1


def test_learned_search_case_uses_persisted_encoder_state(tmp_path: Path) -> None:
    learned_index = _learned_index(tmp_path, _cases())
    output_path = tmp_path / "case_embedding_index.learned.npz"
    save_learned_embedding_index(output_path, learned_index)
    loaded = load_learned_embedding_index(output_path)

    result = loaded.search_case(_case("external", 1, [0.95, 0.05, 0.0]), top_k=1)[0]

    assert result.case.label_id == 1


def test_learned_evaluation_reports_label_match(tmp_path: Path) -> None:
    learned_index = _learned_index(tmp_path, _cases())

    metrics = evaluate_learned_index(learned_index, limit=None, top_k=1)

    assert metrics.top1_label_match_rate == 1.0


def test_learned_evaluation_skips_singleton_index(tmp_path: Path) -> None:
    learned_index = _learned_index(tmp_path, [_case("only", 1, [1.0, 0.0, 0.0])])

    metrics = evaluate_learned_index(learned_index, limit=None, top_k=1)

    assert metrics.evaluated == 0


def _learned_index(tmp_path: Path, cases: list[CaseFeatures]):
    feature_path = tmp_path / "case_feature_index.npz"
    save_compact_feature_index(feature_path, cases)
    feature_index = load_compact_feature_index(feature_path)
    return build_learned_embedding_index(
        feature_index,
        LearnedEncoderConfig(image_dim=2, timeseries_dim=2, centroid_weight=0.25),
    )


def _cases() -> list[CaseFeatures]:
    return [
        _case("query", 1, [1.0, 0.0, 0.0]),
        _case("near", 1, [0.9, 0.1, 0.0]),
        _case("far", 2, [0.0, 1.0, 0.0]),
        _case("far-near", 2, [0.0, 0.9, 0.1]),
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
