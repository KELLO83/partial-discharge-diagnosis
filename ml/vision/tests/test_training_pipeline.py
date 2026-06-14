from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from ml.vision.src.schema import VisionTrainingConfig
from ml.vision.src.training import run_vision_training


def test_vision_dry_run_resolves_train_prefixed_manifest_paths(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "01.원천데이터" / "VS_정상_고체"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "sample.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
    manifest_path = data_dir / "manifest.csv"
    _write_manifest(manifest_path)

    output_dir = tmp_path / "vision"
    summary_path = run_vision_training(
        VisionTrainingConfig(
            manifest_path=manifest_path,
            output_dir=output_dir,
            sample_size=2,
            dry_run=True,
        )
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "dry_run_ready"
    assert payload["train_rows"] == 1
    assert payload["valid_rows"] == 1


def _write_manifest(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "image_path", "label_id"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample-train",
                "split": "train",
                "image_path": "Train/01.원천데이터/VS_정상_고체/sample.png",
                "label_id": "0",
            }
        )
        writer.writerow(
            {
                "sample_id": "sample-valid",
                "split": "valid",
                "image_path": "Train/01.원천데이터/VS_정상_고체/sample.png",
                "label_id": "0",
            }
        )
