from __future__ import annotations

import json

from ml.vlm.src.prompts import build_prompt_text, build_target_json
from ml.vlm.src.schema import ManifestVlmRow, TimeSeriesContext


def test_build_prompt_omits_leakage_fields_from_prompt() -> None:
    row = ManifestVlmRow(
        sample_id="surface_secret_sample",
        image_path="Train/VS_표면방전/sample.png",
        label_id=2,
        label_name="표면방전",
        equipment_name="25.8kV GIS",
        equipment_rated_voltage="22900V",
        equipment_rated_current="600A",
        insulator_type="고체",
        insulator_name="XLPE",
        sensor_type="HFCT",
        temperature="19",
        humidity="66",
        clearance_distance="1000mm",
        defect_details="label leakage detail",
        defect_nums="2",
        max_discharge_value="999",
    )
    context = TimeSeriesContext(
        sample_id="surface_secret_sample",
        ts_model_name="feature_baseline",
        ts_pred_label_id=2,
        ts_confidence=0.82,
        ts_prob_0=0.01,
        ts_prob_1=0.02,
        ts_prob_2=0.82,
        ts_prob_3=0.10,
        ts_prob_4=0.05,
        rms=0.221,
        std=0.14,
        abs_p99=1.3,
        pulse_rate=0.004,
        spectral_energy=3.2,
    )

    prompt = build_prompt_text(row, context)

    assert "25.8kV GIS" in prompt
    assert "ts_pred_class: 2" in prompt
    assert "surface_secret_sample" not in prompt
    assert "표면방전" not in prompt
    assert "label leakage detail" not in prompt
    assert "999" not in prompt
    assert "Train/VS_" not in prompt


def test_build_target_json_is_parseable_diagnosis_schema() -> None:
    row = ManifestVlmRow(
        sample_id="sample-1",
        image_path="Train/sample.png",
        label_id=3,
        label_name="코로나방전",
        equipment_name="GIS",
        equipment_rated_voltage="22900V",
        equipment_rated_current="600A",
        insulator_type="기체",
        insulator_name="SF6",
        sensor_type="UHF",
        temperature="20",
        humidity="60",
        clearance_distance="700mm",
        defect_details="",
        defect_nums="",
        max_discharge_value="",
    )

    target = json.loads(build_target_json(row))

    assert target["label_id"] == 3
    assert target["diagnosis"] == "코로나방전"
    assert set(target) == {
        "label_id",
        "diagnosis",
        "risk_level",
        "reason",
        "recommended_action",
    }
