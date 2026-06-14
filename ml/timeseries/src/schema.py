"""Project-wide constants for partial discharge classification."""

from __future__ import annotations

from dataclasses import dataclass


LABEL_ID_TO_NAME: dict[int, str] = {
    0: "정상",
    1: "노이즈",
    2: "표면 방전",
    3: "코로나 방전",
    4: "보이드 방전",
}

LABEL_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in LABEL_ID_TO_NAME.items()}

N_CLASSES = 5
DEFAULT_SEQUENCE_LENGTH = 7680
DEFAULT_PSEUDO_CHANNELS = 20


@dataclass(frozen=True)
class TimeSeriesShape:
    """Canonical CSV signal shape for the current Train data."""

    pseudo_channels: int = DEFAULT_PSEUDO_CHANNELS
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH

    @property
    def channel_first(self) -> tuple[int, int]:
        return (self.pseudo_channels, self.sequence_length)

    @property
    def time_first(self) -> tuple[int, int]:
        return (self.sequence_length, self.pseudo_channels)
