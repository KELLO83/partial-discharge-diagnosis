"""PyTorch runtime helpers for precision, compile, and SDPA diagnostics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

SDPA_PROBE_BATCH_SIZE = 1
SDPA_PROBE_SEQ_LEN = 128
SDPA_PROBE_NUM_HEADS = 4
SDPA_PROBE_HEAD_DIM = 64
SDPA_PROFILER_EVENT_LIMIT = 20


@dataclass(frozen=True, slots=True)
class CompileConfig:
    enabled: bool = False
    mode: str = "default"


@dataclass(frozen=True, slots=True)
class SdpaProbeConfig:
    device: torch.device | str = "cuda"
    dtype: torch.dtype = torch.float16
    seq_len: int = SDPA_PROBE_SEQ_LEN
    num_heads: int = SDPA_PROBE_NUM_HEADS
    head_dim: int = SDPA_PROBE_HEAD_DIM


def autocast_dtype(mixed_precision: str) -> torch.dtype | None:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "off":
        return None
    raise ValueError(f"Unsupported mixed precision mode: {mixed_precision}")


def autocast_enabled(mixed_precision: str, device: torch.device) -> bool:
    return device.type == "cuda" and autocast_dtype(mixed_precision) is not None


def maybe_compile_model(
    model: nn.Module,
    config: CompileConfig,
    logger: logging.Logger,
) -> tuple[nn.Module, dict[str, Any]]:
    report: dict[str, Any] = {
        "enabled": config.enabled,
        "applied": False,
        "mode": config.mode,
        "error": "",
    }
    if not config.enabled:
        return model, report
    if not hasattr(torch, "compile"):
        report["error"] = "torch.compile is not available in this PyTorch build."
        logger.warning(report["error"])
        return model, report
    try:
        compiled = torch.compile(model, mode=config.mode)
    except Exception as exc:  # pragma: no cover - backend/environment dependent
        report["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("torch.compile failed; continuing with eager model: %s", report["error"])
        return model, report
    report["applied"] = True
    logger.info("torch.compile enabled: mode=%s", config.mode)
    return compiled, report


def get_sdpa_backend_report(config: SdpaProbeConfig | None = None) -> dict[str, Any]:
    probe_config = config or SdpaProbeConfig()
    run_device = torch.device(probe_config.device)
    report: dict[str, Any] = {
        "device": str(run_device),
        "dtype": str(probe_config.dtype).replace("torch.", ""),
        "probe_shape": {
            "batch": SDPA_PROBE_BATCH_SIZE,
            "heads": probe_config.num_heads,
            "seq_len": probe_config.seq_len,
            "head_dim": probe_config.head_dim,
        },
        "enabled_backends": {},
        "can_use_flash_attention": None,
        "can_use_efficient_attention": None,
        "selected_backend": "unavailable",
        "profiler_events": [],
        "error": "",
    }
    if not torch.cuda.is_available() or run_device.type != "cuda":
        report["selected_backend"] = "math"
        return report

    report["enabled_backends"] = {
        "flash": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math": bool(torch.backends.cuda.math_sdp_enabled()),
        "cudnn": bool(torch.backends.cuda.cudnn_sdp_enabled())
        if hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
        else None,
    }
    try:
        probe_shape = (
            SDPA_PROBE_BATCH_SIZE,
            probe_config.num_heads,
            probe_config.seq_len,
            probe_config.head_dim,
        )
        query = torch.randn(probe_shape, device=run_device, dtype=probe_config.dtype)
        key = torch.randn(probe_shape, device=run_device, dtype=probe_config.dtype)
        value = torch.randn(probe_shape, device=run_device, dtype=probe_config.dtype)
        params = torch.nn.attention.SDPAParams(query, key, value, None, 0.0, False, False)
        report["can_use_flash_attention"] = bool(torch.nn.attention.can_use_flash_attention(params, False))
        report["can_use_efficient_attention"] = bool(torch.nn.attention.can_use_efficient_attention(params, False))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], acc_events=True) as profiler:
            torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
            torch.cuda.synchronize(run_device)
        events = sorted(
            {
                event.key
                for event in profiler.key_averages()
                if any(
                    marker in event.key.lower()
                    for marker in ("attention", "sdp", "flash", "efficient", "fmha", "cutlass")
                )
            }
        )
        report["profiler_events"] = events[:SDPA_PROFILER_EVENT_LIMIT]
        report["selected_backend"] = _infer_sdpa_backend(events)
    except Exception as exc:  # pragma: no cover - profiler/kernel availability dependent
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def log_sdpa_backend_report(
    logger: logging.Logger,
    label: str,
    config: SdpaProbeConfig | None = None,
) -> dict[str, Any]:
    report = get_sdpa_backend_report(config)
    logger.info(
        "%s SDPA backend probe: selected_backend=%s enabled=%s can_flash=%s can_efficient=%s events=%s error=%s",
        label,
        report["selected_backend"],
        report["enabled_backends"],
        report["can_use_flash_attention"],
        report["can_use_efficient_attention"],
        report["profiler_events"],
        report["error"],
    )
    return report


def _infer_sdpa_backend(events: list[str]) -> str:
    lowered = " ".join(events).lower()
    if "flash" in lowered:
        return "flash_attention"
    if "efficient" in lowered or "fmha" in lowered or "cutlass" in lowered:
        return "efficient_attention"
    if "cudnn" in lowered:
        return "cudnn_attention"
    if "math" in lowered or "scaled_dot_product" in lowered or "sdp" in lowered:
        return "math"
    return "unknown"
