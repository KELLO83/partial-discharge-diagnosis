from __future__ import annotations

from pathlib import Path

from prpd_similarity_retrieval.batch_evaluation import evaluate_compact_index
from prpd_similarity_retrieval.compact_index import load_compact_feature_index, save_compact_feature_index
from prpd_similarity_retrieval.models import CaseFeatures


def test_batch_evaluation_matches_expected_label_rates(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        [
            _case("query", 1, [1.0, 0.0]),
            _case("near", 1, [0.9, 0.1]),
            _case("far", 2, [0.0, 1.0]),
            _case("far-near", 2, [0.1, 0.9]),
        ],
    )

    metrics = evaluate_compact_index(
        index,
        limit=None,
        top_k=1,
        use_query_label=False,
        metadata_baseline=False,
        batch_size=2,
    )

    assert metrics.top1_label_match_rate == 1.0


def test_batch_metadata_baseline_runs_without_feature_scores(tmp_path: Path) -> None:
    index = _index(tmp_path, [_case("query", 1, [1.0, 0.0]), _case("near", 1, [0.9, 0.1]), _case("far", 2, [0.0, 1.0])])

    metrics = evaluate_compact_index(
        index,
        limit=None,
        top_k=1,
        use_query_label=False,
        metadata_baseline=True,
        batch_size=2,
    )

    assert metrics.evaluated == 3


def test_batch_evaluation_skips_query_without_candidates(tmp_path: Path) -> None:
    index = _index(tmp_path, [_case("only", 1, [1.0, 0.0])])

    metrics = evaluate_compact_index(index, limit=None, top_k=1, use_query_label=False, metadata_baseline=False)

    assert metrics.evaluated == 0


def test_batch_evaluation_returns_label_breakdown(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        [
            _case("query", 1, [1.0, 0.0]),
            _case("near", 1, [0.9, 0.1]),
            _case("far", 2, [0.0, 1.0]),
            _case("far-near", 2, [0.1, 0.9]),
        ],
    )

    metrics = evaluate_compact_index(index, limit=None, top_k=1, use_query_label=False, metadata_baseline=False, breakdown_fields=("label_name",))

    assert metrics.to_dict()["breakdowns"]["label_name"][0]["evaluated"] == 2


def test_batch_evaluation_returns_metadata_breakdown(tmp_path: Path) -> None:
    index = _index(tmp_path, [_case("query", 1, [1.0, 0.0]), _case("near", 1, [0.9, 0.1])])

    metrics = evaluate_compact_index(index, limit=None, top_k=1, use_query_label=False, metadata_baseline=False, breakdown_fields=("sensor_type",))

    assert metrics.to_dict()["breakdowns"]["sensor_type"][0]["value"] == "HFCT"


def _index(tmp_path: Path, cases: list[CaseFeatures]):
    index_path = tmp_path / "case_feature_index.npz"
    save_compact_feature_index(index_path, cases)
    return load_compact_feature_index(index_path)


def _case(sample_id: str, label_id: int, image_vector: list[float]) -> CaseFeatures:
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
            "clearance_distance": "1000mm",
            "equipment_rated_voltage": "22900V",
            "equipment_rated_current": "268A",
        },
        image_vector=image_vector,
        timeseries_vector=image_vector,
    )
