from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_ROOT = PROJECT_ROOT / "service" / "uploads"


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    path: Path
    sha256: str
    size_bytes: int
    content_type: str | None


def store_upload(content: bytes, diagnosis_id: str, filename: str, content_type: str | None) -> StoredArtifact:
    target_dir = (UPLOAD_ROOT / diagnosis_id).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = (target_dir / _safe_filename(filename)).resolve()
    if not _is_inside(target_path, target_dir):
        raise ValueError("upload path escaped diagnosis upload directory")
    target_path.write_bytes(content)
    return StoredArtifact(
        path=target_path,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        content_type=content_type,
    )


def _safe_filename(filename: str) -> str:
    candidate = Path(filename).name.strip()
    if candidate == "":
        raise ValueError("filename is empty")
    return candidate


def _is_inside(path: Path, root: Path) -> bool:
    return root == path or root in path.parents
