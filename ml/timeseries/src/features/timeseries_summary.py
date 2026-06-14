"""Shared summary features for CSV time-series diagnosis context."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

PULSE_THRESHOLD_STD_FACTOR = 3.0


@dataclass(frozen=True, slots=True)
class TimeSeriesSummary:
    rms: float
    std: float
    abs_p99: float
    pulse_rate: float
    spectral_energy: float

    def to_float_dict(self) -> dict[str, float]:
        return asdict(self)


def summarize_signal(signal: np.ndarray) -> TimeSeriesSummary:
    values = np.asarray(signal, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D signal array, got shape={values.shape}")
    abs_signal = np.abs(values)
    threshold = float(np.mean(abs_signal) + PULSE_THRESHOLD_STD_FACTOR * np.std(abs_signal))
    spectrum = np.fft.rfft(values, axis=1)
    return TimeSeriesSummary(
        rms=float(np.sqrt(np.mean(np.square(values)))),
        std=float(np.std(values)),
        abs_p99=float(np.percentile(abs_signal, 99)),
        pulse_rate=float(np.mean(abs_signal > threshold)),
        spectral_energy=float(np.mean(np.square(np.abs(spectrum)))),
    )


def format_summary_for_csv(summary: TimeSeriesSummary) -> dict[str, str]:
    return {key: f"{value:.8g}" for key, value in summary.to_float_dict().items()}
