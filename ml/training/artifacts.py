from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def timestamped_model_dir(root: Path, model_name: str, started_at: datetime | None = None) -> Path:
    timestamp = (started_at or datetime.now()).strftime(RUN_TIMESTAMP_FORMAT)
    return root / safe_model_dirname(model_name) / timestamp


def safe_model_dirname(model_name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.strip())
    return name.strip("._-") or "model"


def relative_artifact_path(base_dir: Path, artifact_path: Path) -> str:
    try:
        return artifact_path.relative_to(base_dir).as_posix()
    except ValueError:
        return str(artifact_path)
