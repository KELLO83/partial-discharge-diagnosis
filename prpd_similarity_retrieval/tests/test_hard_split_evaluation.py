from __future__ import annotations

from pathlib import Path

from prpd_similarity_retrieval.compact_index import load_compact_feature_index, save_compact_feature_index
from prpd_similarity_retrieval.hard_split_evaluation import (
    evaluate_hard_split_report,
    evaluate_feature_hard_split,
    evaluate_learned_hard_split_report,
    evaluate_prototype_hard_split,
    evaluate_prototype_hard_split_report,
    hard_split_failures_to_markdown,
    hard_split_report_to_markdown,
    holdout_values_for_split,
    learned_hard_split_report_to_markdown,
    prototype_hard_split_report_to_markdown,
    sample_feature_hard_split_failures,
    select_hard_split,
)
from prpd_similarity_retrieval.learned_encoder import LearnedEncoderConfig
from prpd_similarity_retrieval.models import CaseFeatures
from prpd_similarity_retrieval.prototype_encoder import PrototypeEncoderConfig


def test_select_hard_split_defaults_to_largest_non_global_group(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        [
            _case("train", 1, [1.0, 0.0], "A"),
            _case("holdout-1", 1, [1.0, 0.0], "B"),
            _case("holdout-2", 2, [0.0, 1.0], "B"),
        ],
    )

    selection = select_hard_split(index, "equipment_name")

    assert selection.holdout_values == ("B",)
    assert selection.train_count == 1
    assert selection.query_count == 2


def test_feature_hard_split_uses_only_train_candidates(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        [
            _case("train-wrong", 2, [0.0, 1.0], "train"),
            _case("holdout-query", 1, [1.0, 0.0], "holdout"),
            _case("holdout-twin", 1, [1.0, 0.0], "holdout"),
        ],
    )

    metrics = evaluate_feature_hard_split(index, split_field="equipment_name", holdout_values=("holdout",), top_k=1)

    assert metrics.train_count == 1
    assert metrics.query_count == 2
    assert metrics.evaluated == 2
    assert metrics.top1_label_match_rate == 0.0


def test_feature_hard_split_finds_same_label_train_neighbor(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        [
            _case("train-good", 1, [1.0, 0.0], "train"),
            _case("train-bad", 2, [0.0, 1.0], "train"),
            _case("holdout-query", 1, [0.95, 0.05], "holdout"),
        ],
    )

    metrics = evaluate_feature_hard_split(index, split_field="equipment_name", holdout_values=("holdout",), top_k=1)

    assert metrics.top1_label_match_rate == 1.0
    assert metrics.topk_label_match_rate == 1.0


def test_prototype_hard_split_fits_on_train_and_evaluates_holdout(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        [
            _case("train-good", 1, [1.0, 0.0], "train"),
            _case("train-bad", 2, [0.0, 1.0], "train"),
            _case("holdout-query", 1, [0.95, 0.05], "holdout"),
        ],
    )

    metrics = evaluate_prototype_hard_split(
        index,
        PrototypeEncoderConfig(image_dim=8, timeseries_dim=8, centroid_weight=0.4, random_seed=11),
        split_field="equipment_name",
        holdout_values=("holdout",),
        top_k=1,
    )

    assert metrics.retrieval_mode == "prototype_encoder"
    assert metrics.train_count == 2
    assert metrics.query_count == 1
    assert metrics.top1_label_match_rate == 1.0


def test_holdout_values_for_split_sorts_by_query_count_then_name(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        [
            _case("b-1", 1, [1.0, 0.0], "B"),
            _case("a-1", 1, [1.0, 0.0], "A"),
            _case("a-2", 2, [0.0, 1.0], "A"),
            _case("c-1", 1, [1.0, 0.0], "C"),
            _case("c-2", 2, [0.0, 1.0], "C"),
            _case("c-3", 2, [0.0, 1.0], "C"),
        ],
    )

    holdout_values = holdout_values_for_split(index, "equipment_name", min_query_count=2, max_holdouts=2)

    assert holdout_values == ["C", "A"]


def test_hard_split_report_outputs_comparisons_and_markdown(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        [
            _case("a-good", 1, [1.0, 0.0], "A"),
            _case("a-bad", 2, [0.0, 1.0], "A"),
            _case("b-good", 1, [0.9, 0.1], "B"),
            _case("b-bad", 2, [0.1, 0.9], "B"),
        ],
    )

    report = evaluate_hard_split_report(
        index,
        split_field="equipment_name",
        max_holdouts=2,
        limit_per_holdout=1,
        top_k=1,
        include_prototype=True,
        prototype_config=PrototypeEncoderConfig(image_dim=8, timeseries_dim=8, centroid_weight=0.25, random_seed=13),
    )
    markdown = hard_split_report_to_markdown(report)

    assert report.to_dict()["holdout_count"] == 2
    assert "prototype_encoder" in report.to_dict()["comparisons"][0]
    assert "|Prototype top-1|Prototype top-1|" not in markdown
    assert "Prototype top-1" in markdown
    assert "|A|" in markdown


def test_prototype_hard_split_report_outputs_metrics_and_markdown(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        [
            _case("a-good", 1, [1.0, 0.0], "A"),
            _case("a-bad", 2, [0.0, 1.0], "A"),
            _case("b-good", 1, [0.9, 0.1], "B"),
            _case("b-bad", 2, [0.1, 0.9], "B"),
        ],
    )

    report = evaluate_prototype_hard_split_report(
        index,
        PrototypeEncoderConfig(image_dim=8, timeseries_dim=8, centroid_weight=0.25, random_seed=13),
        split_field="equipment_name",
        max_holdouts=2,
        limit_per_holdout=1,
        top_k=1,
    )
    markdown = prototype_hard_split_report_to_markdown(report)
    payload = report.to_dict()

    assert payload["retrieval_mode"] == "prototype_encoder"
    assert payload["holdout_count"] == 2
    assert payload["metrics"][0]["retrieval_mode"] == "prototype_encoder"
    assert "Prototype top-1" in markdown
    assert "|A|" in markdown


def test_learned_hard_split_report_outputs_metrics_and_markdown(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        [
            _case("a-good", 1, [1.0, 0.0, 0.0], "A"),
            _case("a-bad", 2, [0.0, 1.0, 0.0], "A"),
            _case("b-good", 1, [0.9, 0.1, 0.0], "B"),
            _case("b-bad", 2, [0.0, 0.9, 0.1], "B"),
        ],
    )

    report = evaluate_learned_hard_split_report(
        index,
        LearnedEncoderConfig(image_dim=2, timeseries_dim=2, centroid_weight=0.25),
        split_field="equipment_name",
        max_holdouts=2,
        limit_per_holdout=1,
        top_k=1,
    )
    markdown = learned_hard_split_report_to_markdown(report)
    payload = report.to_dict()

    assert payload["retrieval_mode"] == "learned_projection_encoder"
    assert payload["holdout_count"] == 2
    assert payload["metrics"][0]["retrieval_mode"] == "learned_projection_encoder"
    assert "Learned top-1" in markdown
    assert "|A|" in markdown


def test_sample_feature_hard_split_failures_returns_topk_misses(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        [
            _case("train-wrong", 2, [0.0, 1.0], "train"),
            _case("holdout-miss", 1, [0.0, 1.0], "holdout"),
            _case("holdout-twin", 1, [0.0, 1.0], "holdout"),
        ],
    )

    sample = sample_feature_hard_split_failures(
        index,
        split_field="equipment_name",
        holdout_values=("holdout",),
        top_k=1,
        max_failures=1,
    )

    assert sample.to_dict()["failure_count"] == 1
    assert sample.failures[0].query_sample_id == "holdout-miss"
    assert sample.failures[0].neighbors[0].sample_id == "train-wrong"
    assert sample.failures[0].neighbors[0].label_id == 2


def test_sample_feature_hard_split_failures_skips_topk_label_hits(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        [
            _case("train-good", 1, [1.0, 0.0], "train"),
            _case("train-bad", 2, [0.0, 1.0], "train"),
            _case("holdout-hit", 1, [1.0, 0.0], "holdout"),
        ],
    )

    sample = sample_feature_hard_split_failures(index, split_field="equipment_name", holdout_values=("holdout",), top_k=1)

    assert sample.failures == []


def test_hard_split_failures_markdown_includes_query_and_neighbors(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        [
            _case("train-wrong", 2, [0.0, 1.0], "train"),
            _case("holdout-miss", 1, [0.0, 1.0], "holdout"),
        ],
    )
    sample = sample_feature_hard_split_failures(index, split_field="equipment_name", holdout_values=("holdout",), top_k=1)

    markdown = hard_split_failures_to_markdown(sample)

    assert "holdout-miss" in markdown
    assert "train-wrong" in markdown
    assert "label-2" in markdown


def _index(tmp_path: Path, cases: list[CaseFeatures]):
    index_path = tmp_path / "case_feature_index.npz"
    save_compact_feature_index(index_path, cases)
    return load_compact_feature_index(index_path)


def _case(sample_id: str, label_id: int, vector: list[float], equipment_name: str) -> CaseFeatures:
    return CaseFeatures(
        sample_id=sample_id,
        label_id=label_id,
        label_name=f"label-{label_id}",
        image_path=f"{sample_id}.png",
        timeseries_path=f"{sample_id}.csv",
        metadata={
            "equipment_name": equipment_name,
            "sensor_type": "HFCT",
            "insulator_type": "solid",
            "clearance_distance": "1000mm",
            "equipment_rated_voltage": "22900V",
            "equipment_rated_current": "268A",
        },
        image_vector=vector,
        timeseries_vector=vector,
    )
