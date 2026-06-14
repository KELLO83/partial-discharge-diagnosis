from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

from prpd_similarity_retrieval.models import CaseFeatures, CaseRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_ROOT / "manifest.csv"
IMAGE_SIZE = 32
PROFILE_BINS = 16
TIMESERIES_HISTOGRAM_BINS = 12
EPSILON = 1e-9

METADATA_FIELDS = (
    "equipment_name",
    "equipment_rated_voltage",
    "equipment_rated_current",
    "equipment_type",
    "insulator_type",
    "insulator_name",
    "sensor_type",
    "clearance_distance",
)


def load_manifest_cases(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    data_root: Path = DEFAULT_DATA_ROOT,
    limit: int | None = None,
) -> list[CaseRecord]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        cases = [_row_to_case(row, data_root) for row in rows]
    return cases if limit is None else cases[:limit]


def extract_case_features(case: CaseRecord) -> CaseFeatures:
    image_vector = extract_image_vector(case.image_path) if case.image_path is not None else None
    timeseries_vector = extract_timeseries_vector(case.timeseries_path) if case.timeseries_path is not None else None
    return CaseFeatures(
        sample_id=case.sample_id,
        label_id=case.label_id,
        label_name=case.label_name,
        image_path=str(case.image_path) if case.image_path is not None else None,
        timeseries_path=str(case.timeseries_path) if case.timeseries_path is not None else None,
        metadata=case.metadata,
        image_vector=image_vector,
        timeseries_vector=timeseries_vector,
    )


def extract_image_vector(image_path: Path) -> list[float] | None:
    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as image:
            grayscale = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
            pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
    except OSError:
        return None

    darkness = 1.0 - pixels
    thumbnail = _safe_standardize(darkness).reshape(-1)
    horizontal_profile = _bin_profile(darkness.mean(axis=0), PROFILE_BINS)
    vertical_profile = _bin_profile(darkness.mean(axis=1), PROFILE_BINS)
    quadrant_energy = _quadrant_energy(darkness)
    descriptors = _image_descriptors(darkness)
    return _normalize_vector(
        [
            *thumbnail.tolist(),
            *horizontal_profile,
            *vertical_profile,
            *quadrant_energy,
            *descriptors,
        ]
    )


def extract_timeseries_vector(timeseries_path: Path) -> list[float] | None:
    values = _read_numeric_csv(timeseries_path)
    if values is None or values.size == 0:
        return None

    centered = values - float(np.mean(values))
    std = float(np.std(centered))
    max_abs = float(np.max(np.abs(centered)))
    rms = float(math.sqrt(float(np.mean(np.square(centered)))))
    abs_values = np.abs(centered)
    histogram = _histogram(abs_values / (max_abs + EPSILON), TIMESERIES_HISTOGRAM_BINS)
    spectrum = np.abs(np.fft.rfft(centered))
    spectral_energy = float(np.sum(np.square(spectrum)))
    high_frequency_ratio = _high_frequency_ratio(spectrum)
    spectral_centroid = _spectral_centroid(spectrum)
    pulse_rate = float(np.mean(abs_values > (float(np.mean(abs_values)) + (2.5 * float(np.std(abs_values))))))
    core_features = [
        _bounded_ratio(std, rms),
        _bounded_ratio(max_abs, rms),
        _bounded_ratio(float(np.percentile(abs_values, 95)), max_abs),
        _bounded_ratio(float(np.percentile(abs_values, 99)), max_abs),
        min(pulse_rate * 10.0, 1.0),
        min(math.log1p(spectral_energy) / 20.0, 1.0),
        high_frequency_ratio,
        spectral_centroid,
    ]
    return _normalize_vector([*core_features, *histogram])


def _row_to_case(row: dict[str, str], data_root: Path) -> CaseRecord:
    return CaseRecord(
        sample_id=row["sample_id"],
        label_id=_optional_int(row.get("label_id")),
        label_name=row.get("label_name", ""),
        image_path=_resolve_dataset_path(row.get("image_path", ""), data_root),
        timeseries_path=_resolve_dataset_path(row.get("timeseries_path", ""), data_root),
        metadata={field: row.get(field, "") for field in METADATA_FIELDS},
    )


def _resolve_dataset_path(raw_path: str, data_root: Path) -> Path | None:
    if raw_path.strip() == "":
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    parts = path.parts
    if parts and parts[0].lower() == "train":
        return data_root.joinpath(*parts[1:])
    return PROJECT_ROOT / path


def _optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_numeric_csv(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except (OSError, ValueError):
        try:
            data = np.genfromtxt(path, delimiter=",", dtype=np.float32)
        except (OSError, ValueError):
            return None
    values = np.asarray(data, dtype=np.float32).reshape(-1)
    finite_values = values[np.isfinite(values)]
    return finite_values if finite_values.size > 0 else None


def _safe_standardize(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < EPSILON:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _bin_profile(values: np.ndarray, bins: int) -> list[float]:
    chunks = np.array_split(values, bins)
    return [float(np.mean(chunk)) if chunk.size > 0 else 0.0 for chunk in chunks]


def _quadrant_energy(values: np.ndarray) -> list[float]:
    height, width = values.shape
    halves = [
        values[: height // 2, : width // 2],
        values[: height // 2, width // 2 :],
        values[height // 2 :, : width // 2],
        values[height // 2 :, width // 2 :],
    ]
    total = float(np.sum(values)) + EPSILON
    return [float(np.sum(part) / total) for part in halves]


def _image_descriptors(values: np.ndarray) -> list[float]:
    active = values > (float(np.mean(values)) + float(np.std(values)))
    active_ratio = float(np.mean(active))
    entropy = _entropy(values)
    compactness = _compactness(active)
    return [active_ratio, entropy, compactness, float(np.mean(values)), float(np.std(values))]


def _entropy(values: np.ndarray) -> float:
    histogram, _ = np.histogram(values.reshape(-1), bins=16, range=(0.0, 1.0), density=False)
    probabilities = histogram.astype(np.float64)
    probabilities = probabilities / max(float(np.sum(probabilities)), EPSILON)
    nonzero = probabilities[probabilities > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)) / math.log2(16))


def _compactness(active: np.ndarray) -> float:
    coordinates = np.argwhere(active)
    if coordinates.size == 0:
        return 0.0
    span = np.ptp(coordinates, axis=0) + 1
    bounding_area = float(span[0] * span[1])
    return float(min(1.0, coordinates.shape[0] / max(bounding_area, 1.0)))


def _histogram(values: np.ndarray, bins: int) -> list[float]:
    histogram, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    total = float(np.sum(histogram))
    if total < EPSILON:
        return [0.0] * bins
    return [float(item / total) for item in histogram]


def _high_frequency_ratio(spectrum: np.ndarray) -> float:
    if spectrum.size <= 2:
        return 0.0
    cutoff = max(1, spectrum.size // 3)
    total = float(np.sum(spectrum)) + EPSILON
    return float(np.sum(spectrum[cutoff:]) / total)


def _spectral_centroid(spectrum: np.ndarray) -> float:
    if spectrum.size <= 1:
        return 0.0
    weights = np.arange(spectrum.size, dtype=np.float32)
    return float(np.sum(weights * spectrum) / ((spectrum.size - 1) * (np.sum(spectrum) + EPSILON)))


def _bounded_ratio(numerator: float, denominator: float) -> float:
    return float(min(max(numerator / (denominator + EPSILON), 0.0), 1.0))


def _normalize_vector(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    array[~np.isfinite(array)] = 0.0
    norm = float(np.linalg.norm(array))
    if norm < EPSILON:
        return [0.0 for _ in values]
    return [round(float(item / norm), 8) for item in array]
