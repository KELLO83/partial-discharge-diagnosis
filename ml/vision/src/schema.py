from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


DEFAULT_IMAGE_SIZE = 224
DEFAULT_NUM_CLASSES = 5
DEFAULT_VALID_RATIO = 0.2
DEFAULT_SEED = 42

PD_LABELS_KO: dict[int, str] = {
    0: "정상",
    1: "노이즈",
    2: "표면방전",
    3: "코로나방전",
    4: "보이드방전",
}

NORMALIZE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
NORMALIZE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class VisionTrainingConfig:
    manifest_path: Path = Path("data/manifest.csv")
    output_dir: Path = Path("artifacts/models/vision")
    model_name: str = "small_prpd_cnn"
    image_size: int = DEFAULT_IMAGE_SIZE
    sample_size: int | None = None
    valid_ratio: float = DEFAULT_VALID_RATIO
    seed: int = DEFAULT_SEED
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-3
    num_workers: int = 0
    device: str = "auto"
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class VisionManifestSplit:
    train_rows: "pd.DataFrame"
    valid_rows: "pd.DataFrame"
    split_type: str
