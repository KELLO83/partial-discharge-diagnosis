"""Single-model training runner for partial discharge classification."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ml.timeseries.src.data.loader import PartialDischargeDataset, load_manifest, make_stratified_split
from ml.timeseries.src.eval.metrics import classification_metrics
from ml.timeseries.src.experiments.logger import append_experiment_result
from ml.timeseries.src.models.registry import create_model
from ml.timeseries.src.torch_runtime import (
    CompileConfig,
    SdpaProbeConfig,
    autocast_dtype,
    autocast_enabled,
    log_sdpa_backend_report,
    maybe_compile_model,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_GPU_MEMORY_TARGET = 0.90
DEFAULT_MAX_AUTO_BATCH_SIZE = 256


def _device(device: str = "cuda") -> torch.device:
    if device != "cuda":
        LOGGER.info("CPU training is disabled for this project. Requested device=%s.", device)
        raise RuntimeError("Only CUDA GPU training is supported.")
    if not torch.cuda.is_available():
        LOGGER.info("CUDA GPU is not available. CPU training is disabled for this project.")
        raise RuntimeError("CUDA GPU is required for training.")
    requested = torch.device(device)
    return requested


def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "valid",
    mixed_precision: str = "fp16",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    with torch.no_grad():
        progress = tqdm(loader, desc=desc, leave=False)
        for x, y in progress:
            with torch.amp.autocast(
                device_type=device.type,
                dtype=autocast_dtype(mixed_precision),
                enabled=autocast_enabled(mixed_precision, device),
            ):
                logits = model(x.to(device, non_blocking=True))
            y_true.append(y.cpu().numpy())
            y_pred.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def _clear_cuda_state(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def _try_train_step(
    model: nn.Module,
    sample_shape: tuple[int, ...],
    batch_size: int,
    device: torch.device,
    criterion: nn.Module,
    mixed_precision: str,
) -> int:
    """Run one synthetic forward/backward step and return peak reserved CUDA bytes."""
    _clear_cuda_state(device)
    model.train()
    model.zero_grad(set_to_none=True)
    x = torch.randn((batch_size, *sample_shape), device=device)
    y = torch.zeros(batch_size, dtype=torch.long, device=device)
    with torch.amp.autocast(
        device_type=device.type,
        dtype=autocast_dtype(mixed_precision),
        enabled=autocast_enabled(mixed_precision, device),
    ):
        loss = criterion(model(x), y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    del x, y, loss
    _clear_cuda_state(device)
    return int(peak_reserved)


def _resolve_batch_size(
    model: nn.Module,
    sample_shape: tuple[int, ...],
    train_rows: int,
    device: torch.device,
    criterion: nn.Module,
    batch_size: int | None,
    auto_batch_start_size: int,
    target_gpu_memory_utilization: float,
    max_auto_batch_size: int,
    mixed_precision: str,
) -> tuple[int, bool, dict[str, Any]]:
    """Resolve a manual or auto-tuned batch size for a single CUDA training job."""
    if batch_size is not None:
        return batch_size, False, {"mode": "manual"}
    if device.type != "cuda":
        raise RuntimeError("Auto batch sizing requires CUDA.")
    if train_rows <= 0:
        raise ValueError("Training split is empty; cannot resolve batch size.")

    props = torch.cuda.get_device_properties(device)
    total_bytes = int(props.total_memory)
    target_bytes = int(total_bytes * target_gpu_memory_utilization)
    start_batch = max(1, min(auto_batch_start_size, train_rows, max_auto_batch_size))
    LOGGER.info(
        "Auto batch sizing started: target_gpu_memory_utilization=%.2f start_batch=%s max_auto_batch_size=%s train_rows=%s",
        target_gpu_memory_utilization,
        start_batch,
        max_auto_batch_size,
        train_rows,
    )

    probe_batch = start_batch
    while probe_batch >= 1:
        try:
            peak_reserved = _try_train_step(model, sample_shape, probe_batch, device, criterion, mixed_precision)
            break
        except torch.cuda.OutOfMemoryError:
            LOGGER.info("Auto batch probe OOM at batch_size=%s; retrying with half.", probe_batch)
            _clear_cuda_state(device)
            probe_batch //= 2
    else:
        raise RuntimeError("Unable to run even batch_size=1 on CUDA.")

    per_sample_bytes = max(1, peak_reserved // max(1, probe_batch))
    estimated = max(1, target_bytes // per_sample_bytes)
    candidate = min(int(estimated), train_rows, max_auto_batch_size)
    candidate = max(1, candidate)

    while candidate >= 1:
        try:
            peak_reserved = _try_train_step(model, sample_shape, candidate, device, criterion, mixed_precision)
            if peak_reserved <= target_bytes or candidate == 1:
                break
            LOGGER.info(
                "Auto batch candidate exceeded target: batch_size=%s peak=%.2fGB target=%.2fGB; retrying with half.",
                candidate,
                peak_reserved / (1024**3),
                target_bytes / (1024**3),
            )
            candidate //= 2
        except torch.cuda.OutOfMemoryError:
            LOGGER.info("Auto batch candidate OOM at batch_size=%s; retrying with half.", candidate)
            _clear_cuda_state(device)
            candidate //= 2
    if candidate < 1:
        candidate = 1

    utilization = peak_reserved / total_bytes
    report = {
        "mode": "auto",
        "probe_batch_size": probe_batch,
        "target_gpu_memory_utilization": target_gpu_memory_utilization,
        "resolved_batch_size": candidate,
        "peak_reserved_bytes": int(peak_reserved),
        "peak_reserved_gb": round(peak_reserved / (1024**3), 4),
        "gpu_total_gb": round(total_bytes / (1024**3), 4),
        "estimated_peak_memory_utilization": round(utilization, 6),
        "max_auto_batch_size": max_auto_batch_size,
    }
    LOGGER.info(
        "Auto batch sizing resolved: batch_size=%s peak_reserved=%.2fGB total=%.2fGB estimated_utilization=%.3f target=%.2f",
        candidate,
        report["peak_reserved_gb"],
        report["gpu_total_gb"],
        utilization,
        target_gpu_memory_utilization,
    )
    return candidate, True, report


def run_single_experiment(
    model_name: str,
    manifest_path: Path,
    output_path: Path,
    sample_size: int | None = None,
    valid_ratio: float = 0.2,
    seed: int = 42,
    epochs: int = 3,
    batch_size: int | None = None,
    auto_batch_start_size: int = 1,
    target_gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_TARGET,
    max_auto_batch_size: int = DEFAULT_MAX_AUTO_BATCH_SIZE,
    learning_rate: float = 1e-3,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: str = "cuda",
    model_params: dict[str, Any] | None = None,
    mixed_precision: str = "fp16",
    torch_compile: bool = False,
    torch_compile_mode: str = "default",
) -> dict[str, Any]:
    """Run exactly one model experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    LOGGER.info("Loading manifest: %s", manifest_path)
    manifest = load_manifest(manifest_path)
    LOGGER.info("Loaded manifest rows=%s columns=%s", len(manifest), len(manifest.columns))
    split = make_stratified_split(manifest, valid_ratio=valid_ratio, seed=seed, sample_size=sample_size)
    LOGGER.info(
        "Created split: train_rows=%s valid_rows=%s valid_ratio=%s sample_size=%s",
        len(split.train),
        len(split.valid),
        valid_ratio,
        sample_size or "full",
    )
    LOGGER.info("Train label counts: %s", split.train["label_id"].value_counts().sort_index().to_dict())
    LOGGER.info("Valid label counts: %s", split.valid["label_id"].value_counts().sort_index().to_dict())

    model = create_model(model_name, params=model_params)
    model_metadata = {
        "name": model.name,
        "family": model.family,
        "input_layout": model.input_layout,
        "training_mode": model.training_mode,
        "pretrained": model.pretrained,
    }
    run_device = _device(device)
    model.to(run_device)
    LOGGER.info(
        "Created model=%s family=%s layout=%s device=%s mixed_precision=%s torch_compile=%s python=%s executable=%s",
        model_metadata["name"],
        model_metadata["family"],
        model_metadata["input_layout"],
        run_device,
        mixed_precision,
        torch_compile,
        sys.version.split()[0],
        sys.executable,
    )
    sdpa_report = log_sdpa_backend_report(
        LOGGER,
        "time-series training",
        SdpaProbeConfig(device=run_device, dtype=autocast_dtype(mixed_precision) or torch.float32),
    )

    train_ds = PartialDischargeDataset(split.train, layout=model_metadata["input_layout"])
    valid_ds = PartialDischargeDataset(split.valid, layout=model_metadata["input_layout"])
    sample_x, sample_y = train_ds[0]
    criterion = nn.CrossEntropyLoss()
    batch_size, batch_size_auto, batch_size_report = _resolve_batch_size(
        model=model,
        sample_shape=tuple(sample_x.shape),
        train_rows=len(train_ds),
        device=run_device,
        criterion=criterion,
        batch_size=batch_size,
        auto_batch_start_size=auto_batch_start_size,
        target_gpu_memory_utilization=target_gpu_memory_utilization,
        max_auto_batch_size=max_auto_batch_size,
        mixed_precision=mixed_precision,
    )
    LOGGER.info(
        "Dataset config: input_layout=%s sample_shape=%s sample_label=%s normalize=True batch_size=%s batch_size_auto=%s num_workers=%s pin_memory=%s",
        model_metadata["input_layout"],
        tuple(sample_x.shape),
        int(sample_y.item()),
        batch_size,
        batch_size_auto,
        num_workers,
        pin_memory,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model, torch_compile_report = maybe_compile_model(
        model=model,
        config=CompileConfig(enabled=torch_compile, mode=torch_compile_mode),
        logger=LOGGER,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=run_device.type == "cuda" and mixed_precision == "fp16")

    start_train = time.perf_counter()
    LOGGER.info(
        "Training started: epochs=%s batch_size=%s lr=%s mixed_precision=%s grad_scaler=%s",
        epochs,
        batch_size,
        learning_rate,
        mixed_precision,
        scaler.is_enabled(),
    )
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{epochs}", leave=False)
        for x, y in progress:
            x = x.to(run_device, non_blocking=True)
            y = y.to(run_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=run_device.type,
                dtype=autocast_dtype(mixed_precision),
                enabled=autocast_enabled(mixed_precision, run_device),
            ):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")
        LOGGER.info("Epoch %s completed: avg_loss=%.6f", epoch, running_loss / max(1, len(train_loader)))
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    LOGGER.info("Validation prediction started")
    y_true, y_pred = _predict(
        model,
        valid_loader,
        run_device,
        desc=f"{model_name} validation",
        mixed_precision=mixed_precision,
    )
    predict_time = time.perf_counter() - start_predict
    metrics = classification_metrics(y_true, y_pred)
    LOGGER.info(
        "Validation completed in %.3f sec: accuracy=%.6f macro_f1=%.6f rows=%s",
        predict_time,
        metrics.accuracy,
        metrics.macro_f1,
        metrics.n_rows,
    )
    LOGGER.info("Per-class F1: %s", metrics.per_class_f1)
    LOGGER.info("Confusion matrix: %s", metrics.confusion_matrix)

    row = {
        "experiment_id": f"{model_name}_{sample_size or 'full'}_seed{seed}",
        "model_name": model_metadata["name"],
        "model_family": model_metadata["family"],
        "training_mode": model_metadata["training_mode"],
        "pretrained": model_metadata["pretrained"],
        "manifest_path": str(manifest_path),
        "split_type": split.split_type,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "device": str(run_device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "input_layout": model_metadata["input_layout"],
        "sample_size": sample_size or len(manifest),
        "train_rows": len(split.train),
        "valid_rows": len(split.valid),
        "valid_ratio": valid_ratio,
        "split_seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "batch_size_auto": batch_size_auto,
        "batch_size_report": batch_size_report,
        "target_gpu_memory_utilization": target_gpu_memory_utilization,
        "max_auto_batch_size": max_auto_batch_size,
        "learning_rate": learning_rate,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "model_params": model_params or {},
        "mixed_precision": mixed_precision,
        "torch_compile": torch_compile,
        "torch_compile_report": torch_compile_report,
        "sdpa_backend_report": sdpa_report,
        "train_time_sec": round(train_time, 6),
        "predict_time_sec": round(predict_time, 6),
        "valid_accuracy": metrics.accuracy,
        "valid_macro_f1": metrics.macro_f1,
        "valid_weighted_f1": metrics.weighted_f1,
        "valid_balanced_accuracy": metrics.balanced_accuracy,
        "valid_per_class_f1": metrics.per_class_f1,
        "valid_per_class_precision": metrics.per_class_precision,
        "valid_per_class_recall": metrics.per_class_recall,
        "valid_pd_to_normal_error_count": metrics.pd_to_normal_error_count,
        "valid_normal_recall": metrics.per_class_recall[0],
        "valid_noise_recall": metrics.per_class_recall[1],
        "valid_surface_recall": metrics.per_class_recall[2],
        "valid_corona_recall": metrics.per_class_recall[3],
        "valid_void_recall": metrics.per_class_recall[4],
        "valid_confusion_matrix": metrics.confusion_matrix,
    }
    append_experiment_result(output_path, row)
    LOGGER.info("Experiment result saved: %s", output_path)
    return row
