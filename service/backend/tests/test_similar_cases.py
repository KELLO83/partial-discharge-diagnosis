from __future__ import annotations

from pathlib import Path

from service.backend.app.domain.policy import label_name
from service.backend.app.schemas import MetadataInput, TimeSeriesResult, VisionResult
from service.backend.app.domain.similar_cases import DatasetCaseRepository


def test_repository_resolves_train_prefix_and_returns_similar_cases(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    image_path = data_root / "01.원천데이터" / "VS_코로나방전_고체_ACSR-OC" / "case.png"
    csv_path = image_path.with_suffix(".csv")
    json_path = data_root / "02.라벨링데이터" / "VL_코로나방전_고체_ACSR-OC" / "case.json"
    image_path.parent.mkdir(parents=True)
    json_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")
    csv_path.write_text("1,2,3\n", encoding="utf-8")
    json_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "sample_id,json_path,image_path,timeseries_path,label_id,label_name,insulator_type,equipment_name,equipment_rated_voltage,equipment_rated_current,sensor_type,temperature,humidity,clearance_distance,max_discharge_value",
                "case,Train/02.라벨링데이터/VL_코로나방전_고체_ACSR-OC/case.json,Train/01.원천데이터/VS_코로나방전_고체_ACSR-OC/case.png,Train/01.원천데이터/VS_코로나방전_고체_ACSR-OC/case.csv,3,코로나 방전,고체,ACSR-OC,22900V,268A,HFCT,19,66,['1000mm'],82",
            ]
        ),
        encoding="utf-8",
    )
    repository = DatasetCaseRepository(manifest_path=manifest_path, data_root=data_root)

    cases = repository.similar_cases(_metadata(), _timeseries_result(), _vision_result())

    assert cases[0].sample_id == "case"
    assert cases[0].image_url == "/dataset/cases/case/image"
    assert repository.get("case").image_path == image_path


def _metadata() -> MetadataInput:
    return MetadataInput(
        equipment_name="ACSR-OC",
        equipment_type="overhead line",
        equipment_rated_voltage="22900V",
        equipment_rated_current="268A",
        sensor_type="HFCT",
        temperature=19,
        humidity=66,
        insulator_type="고체",
        clearance_distance="1000mm",
    )


def _timeseries_result() -> TimeSeriesResult:
    return TimeSeriesResult(
        model_name="test_ts",
        model_version="test",
        label_id=3,
        label_name=label_name(3),
        confidence=0.9,
        probabilities={"0": 0.02, "1": 0.03, "2": 0.03, "3": 0.9, "4": 0.02},
        features={"abs_p99": 78.0},
    )


def _vision_result() -> VisionResult:
    return VisionResult(
        model_name="test_vision",
        model_version="test",
        label_id=3,
        label_name=label_name(3),
        confidence=0.9,
        probabilities={"0": 0.02, "1": 0.03, "2": 0.03, "3": 0.9, "4": 0.02},
        evidence={"phase_localization_score": 0.8},
    )
