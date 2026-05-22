"""Helpers for loading official or verified external model implementations."""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator


DEFAULT_REPO_PATHS = {
    "TSLIB_REPO": Path("external/Time-Series-Library"),
    "ITRANSFORMER_REPO": Path("external/iTransformer"),
    "TIMEMIXER_REPO": Path("external/TimeMixer"),
    "UNITS_REPO": Path("external/UniTS"),
    "ONE_FITS_ALL_REPO": Path("external/One_Fits_All"),
    "TS2VEC_REPO": Path("external/ts2vec"),
}

TOP_LEVEL_EXTERNAL_PACKAGES = ("models", "layers", "utils", "data_provider", "exp", "model")


def require_module(module_name: str, install_hint: str):
    """Import a required external module with an actionable error message."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(install_hint) from exc


@contextmanager
def prepend_repo_path(env_var: str, install_hint: str) -> Iterator[Path]:
    """Temporarily prepend an official cloned repository path to sys.path."""
    repo_path = os.environ.get(env_var)
    if repo_path:
        path = Path(repo_path).expanduser().resolve()
    elif env_var in DEFAULT_REPO_PATHS:
        path = DEFAULT_REPO_PATHS[env_var].resolve()
    else:
        raise ImportError(install_hint)
    if not path.exists():
        raise ImportError(f"{install_hint} Current/default {env_var} does not exist: {path}")
    sys.path.insert(0, str(path))
    try:
        yield path
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass


def clear_external_module_cache() -> None:
    """Clear top-level module names commonly reused by official time-series repos."""
    for name in list(sys.modules):
        root = name.split(".", 1)[0]
        if root in TOP_LEVEL_EXTERNAL_PACKAGES:
            sys.modules.pop(name, None)


def namespace_config(config: dict[str, object] | None = None, **defaults: object) -> SimpleNamespace:
    """Create a SimpleNamespace config object expected by many official repos."""
    merged = dict(defaults)
    merged.update(config or {})
    return SimpleNamespace(**merged)
