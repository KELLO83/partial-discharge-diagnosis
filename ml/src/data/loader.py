"""Manifest-based dataset loading for partial discharge time-series CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.src.schema import DEFAULT_PSEUDO_CHANNELS, DEFAULT_SEQUENCE_LENGTH

InputLayout = Literal["channel_first", "time_first"]


@dataclass(frozen=True)
class ManifestSplit:
    train: pd.DataFrame
    valid: pd.DataFrame


def load_manifest(path: str | Path = "Train/manifest.csv") -> pd.DataFrame:
    """Load the generated manifest file."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run scripts/build_train_manifest.py first."
        )
    frame = pd.read_csv(manifest_path)
    required = {"timeseries_path", "label_id", "sample_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return frame


def make_stratified_split(
    manifest: pd.DataFrame,
    valid_ratio: float = 0.2,
    seed: int = 42,
    sample_size: int | None = None,
) -> ManifestSplit:
    """Create a deterministic stratified train/validation split from manifest rows."""
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("valid_ratio must be between 0 and 1.")

    frame = manifest.copy()
    if sample_size is not None and sample_size < len(frame):
        sampled_parts = []
        per_class = max(1, sample_size // int(frame["label_id"].nunique()))
        for _, part in frame.groupby("label_id", sort=True):
            sampled_parts.append(part.sample(min(len(part), per_class), random_state=seed))
        frame = pd.concat(sampled_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    for _, part in frame.groupby("label_id", sort=True):
        shuffled = part.sample(frac=1.0, random_state=seed)
        n_valid = max(1, int(round(len(shuffled) * valid_ratio)))
        valid_parts.append(shuffled.iloc[:n_valid])
        train_parts.append(shuffled.iloc[n_valid:])

    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    valid = pd.concat(valid_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return ManifestSplit(train=train, valid=valid)


def read_timeseries_csv(path: str | Path, dtype: np.dtype = np.float32) -> np.ndarray:
    """Read one headerless CSV signal as (pseudo_channel, time)."""
    array = np.loadtxt(path, delimiter=",", dtype=dtype)
    if array.shape != (DEFAULT_PSEUDO_CHANNELS, DEFAULT_SEQUENCE_LENGTH):
        raise ValueError(
            f"Unexpected CSV shape for {path}: {array.shape}. "
            f"Expected {(DEFAULT_PSEUDO_CHANNELS, DEFAULT_SEQUENCE_LENGTH)}."
        )
    return array


class PartialDischargeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch Dataset for CSV-only partial discharge classification."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        layout: InputLayout = "channel_first",
        normalize: bool = True,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.layout = layout
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[index]
        signal = read_timeseries_csv(row["timeseries_path"])
        if self.normalize:
            mean = signal.mean(dtype=np.float64)
            std = signal.std(dtype=np.float64)
            signal = (signal - mean) / max(float(std), 1e-6)
        if self.layout == "time_first":
            signal = signal.T
        elif self.layout != "channel_first":
            raise ValueError(f"Unsupported layout: {self.layout}")
        x = torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32))
        y = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return x, y
