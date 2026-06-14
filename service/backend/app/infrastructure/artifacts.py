from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Collection

PROJECT_ROOT = Path(__file__).resolve().parents[4]
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


def find_uploaded_file(diagnosis_id: str, extensions: Collection[str]) -> Path | None:
    diagnosis_dir = (UPLOAD_ROOT / Path(diagnosis_id).name).resolve()
    upload_root = UPLOAD_ROOT.resolve()
    if not _is_inside(diagnosis_dir, upload_root) or not diagnosis_dir.is_dir():
        return None
    normalized_extensions = {extension.lower() for extension in extensions}
    candidates = sorted(
        path
        for path in diagnosis_dir.iterdir()
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )
    return candidates[0] if candidates else None


def _safe_filename(filename: str) -> str:
    candidate = Path(filename).name.strip()
    if candidate == "":
        raise ValueError("filename is empty")
    return candidate


def _is_inside(path: Path, root: Path) -> bool:
    return root == path or root in path.parents
