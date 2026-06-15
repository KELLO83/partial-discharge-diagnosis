"""Single-model training runner for partial discharge classification."""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ml.timeseries.src.data.loader import PartialDischargeDataset, load_manifest, make_stratified_split
from ml.timeseries.src.eval.metrics import classification_metrics
from ml.timeseries.src.experiments.logger import append_experiment_result
from ml.timeseries.src.models.registry import create_model
from ml.timeseries.src.schema import LABEL_ID_TO_NAME
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
DEFAULT_ARTIFACT_DIR = Path("artifacts/models/time_series")
BEST_CHECKPOINT_FILENAME = "best.pt"
RESUME_CHECKPOINT_FILENAME = "resumet.pt"
TRAIN_SUMMARY_FILENAME = "train_summary.json"
TRAINING_CONFIG_FILENAME = "training_config.json"
PREPROCESSOR_FILENAME = "preprocessor.json"
MODEL_MANIFEST_FILENAME = "model_manifest.json"
TENSORBOARD_DIRNAME = "tensorboard"
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.0
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_SCHEDULER = "onecycle"


def _resolve_device(device: str, allow_cpu: bool = False) -> torch.device:
    if device == "cpu":
        if not allow_cpu:
            LOGGER.info("CPU training is disabled for this project. Requested device=%s.", device)
            raise RuntimeError("Only CUDA GPU training is supported.")
        return torch.device("cpu")
    if device != "cuda":
        raise RuntimeError(f"Unsupported device: {device}. Supported values are cuda|cpu.")
    if not torch.cuda.is_available():
        if allow_cpu:
            return torch.device("cpu")
        LOGGER.info("CUDA GPU is not available. CPU training is disabled for this project.")
        raise RuntimeError("CUDA GPU is required for training.")
    return torch.device("cuda")


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


def _configure_cuda_memory(device: torch.device, gpu_memory_fraction: float) -> None:
    if device.type != "cuda":
        return
    if not 0 < gpu_memory_fraction <= 1:
        raise ValueError("gpu memory fraction must be in (0, 1].")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device_index)


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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        peak_reserved = 0
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
    """Resolve a manual or auto-tuned batch size for a single training run."""
    if batch_size is not None:
        return batch_size, False, {"mode": "manual"}
    if device.type != "cuda":
        raise RuntimeError("Auto batch sizing requires CUDA.")
    if train_rows <= 0:
        raise ValueError("Training split is empty. Cannot resolve batch size.")

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
            peak_reserved = _try_train_step(
                model=model,
                sample_shape=sample_shape,
                batch_size=probe_batch,
                device=device,
                criterion=criterion,
                mixed_precision=mixed_precision,
            )
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
            peak_reserved = _try_train_step(
                model=model,
                sample_shape=sample_shape,
                batch_size=candidate,
                device=device,
                criterion=criterion,
                mixed_precision=mixed_precision,
            )
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

    utilization = peak_reserved / max(1, total_bytes)
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
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: str = "cuda",
    model_params: dict[str, Any] | None = None,
    mixed_precision: str = "fp16",
    torch_compile: bool = False,
    torch_compile_mode: str = "default",
    artifact_dir: Path | None = None,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    scheduler_name: str = DEFAULT_SCHEDULER,
    resume_from: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run exactly one model experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    service_artifact_dir = artifact_dir or DEFAULT_ARTIFACT_DIR
    artifact_dir = service_artifact_dir if dry_run else _timestamped_run_dir(service_artifact_dir, model_name)
    artifact_dir.mkdir(parents=True, exist_ok=True)

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

    training_config = {
        "model_name": model_name,
        "seed": seed,
        "sample_size": sample_size or len(manifest),
        "valid_ratio": valid_ratio,
        "epochs": epochs,
        "batch_size": batch_size,
        "auto_batch_start_size": auto_batch_start_size,
        "target_gpu_memory_utilization": target_gpu_memory_utilization,
        "max_auto_batch_size": max_auto_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "device": device,
        "mixed_precision": mixed_precision,
        "torch_compile": torch_compile,
        "torch_compile_mode": torch_compile_mode,
        "model_params": model_params or {},
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "scheduler": scheduler_name,
        "resume_from": str(resume_from) if resume_from is not None else None,
    }
    (artifact_dir / TRAINING_CONFIG_FILENAME).write_text(
        json.dumps(training_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_dataset = PartialDischargeDataset(split.train, layout=model_metadata["input_layout"])
    valid_dataset = PartialDischargeDataset(split.valid, layout=model_metadata["input_layout"])
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("Train/valid split is empty. Adjust sample-size or valid-ratio.")

    _write_smoke_report(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        output_path=output_path,
        split=split,
        model_name=model_name,
        model_metadata=model_metadata,
        model_params=model_params,
        sample_size=sample_size,
        seed=seed,
        epochs=epochs,
        dry_run=dry_run,
    )

    if dry_run:
        run_device = _resolve_device(device="cpu", allow_cpu=True)
        return _dry_run_experiment(
            model=model,
            model_metadata=model_metadata,
            model_params=model_params,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            manifest_path=manifest_path,
            output_path=output_path,
            split=split,
            split_name=split.split_type,
            sample_size=sample_size,
            seed=seed,
            epochs=epochs,
            run_device=run_device,
            artifact_dir=artifact_dir,
            mixed_precision=mixed_precision,
        )

    run_device = _resolve_device(device)
    _configure_cuda_memory(run_device, target_gpu_memory_utilization)
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

    sample_x, sample_y = train_dataset[0]
    criterion = nn.CrossEntropyLoss()
    batch_size, batch_size_auto, batch_size_report = _resolve_batch_size(
        model=model,
        sample_shape=tuple(sample_x.shape),
        train_rows=len(train_dataset),
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
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
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
    optimizer = _build_optimizer(model, learning_rate, weight_decay)
    scheduler = _build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        learning_rate=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=run_device.type == "cuda" and mixed_precision == "fp16")
    resume_state = _load_resume_state(
        resume_from=resume_from,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=run_device,
    )

    start_train = time.perf_counter()
    LOGGER.info(
        "Training started: epochs=%s batch_size=%s lr=%s mixed_precision=%s grad_scaler=%s artifact_dir=%s",
        epochs,
        batch_size,
        learning_rate,
        mixed_precision,
        scaler.is_enabled(),
        artifact_dir,
    )
    best_metric = float(resume_state.get("best_metric", float("-inf")))
    best_epoch = int(resume_state.get("best_epoch", 0))
    best_metrics = None
    best_predict_time = 0.0
    best_y_true = None
    best_y_pred = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    tensorboard_dir = artifact_dir / TENSORBOARD_DIRNAME
    with SummaryWriter(log_dir=str(tensorboard_dir)) as writer:
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        for epoch in range(start_epoch, epochs + 1):
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
                if scheduler is not None:
                    scheduler.step()
                running_loss += float(loss.item())
                progress.set_postfix(loss=f"{loss.item():.4f}")
            avg_loss = running_loss / max(1, len(train_loader))
            start_predict = time.perf_counter()
            y_true, y_pred = _predict(
                model,
                valid_loader,
                run_device,
                desc=f"{model_name} validation epoch {epoch}",
                mixed_precision=mixed_precision,
            )
            predict_time = time.perf_counter() - start_predict
            metrics = classification_metrics(y_true, y_pred)
            metric = float(metrics.macro_f1)
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("eval/accuracy", metrics.accuracy, epoch)
            writer.add_scalar("eval/macro_f1", metrics.macro_f1, epoch)
            if scheduler is not None:
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
            improved = metric > best_metric + early_stopping_min_delta
            if improved:
                best_metric = metric
                best_epoch = epoch
                best_metrics = metrics
                best_predict_time = predict_time
                best_y_true = y_true
                best_y_pred = y_pred
                stale_epochs = 0
                _write_checkpoint(
                    artifact_dir=artifact_dir,
                    model=model,
                    model_name=model_name,
                    input_layout=model_metadata["input_layout"],
                    model_params=model_params,
                    filename=BEST_CHECKPOINT_FILENAME,
                    epoch=epoch,
                    metric_name="valid_macro_f1",
                    metric_value=metric,
                )
            else:
                stale_epochs += 1
            _write_resume_checkpoint(
                artifact_dir=artifact_dir,
                model=model,
                optimizer=optimizer,
                model_name=model_name,
                input_layout=model_metadata["input_layout"],
                model_params=model_params,
                epoch=epoch,
                best_epoch=best_epoch,
                best_metric=best_metric,
                scheduler=scheduler,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "valid_accuracy": metrics.accuracy,
                    "valid_macro_f1": metrics.macro_f1,
                    "improved": improved,
                }
            )
            LOGGER.info(
                "Epoch %s completed: avg_loss=%.6f valid_accuracy=%.6f valid_macro_f1=%.6f best_epoch=%s stale_epochs=%s",
                epoch,
                avg_loss,
                metrics.accuracy,
                metrics.macro_f1,
                best_epoch,
                stale_epochs,
            )
            if stale_epochs >= early_stopping_patience:
                LOGGER.info(
                    "Early stopping triggered: epoch=%s patience=%s best_epoch=%s best_valid_macro_f1=%.6f",
                    epoch,
                    early_stopping_patience,
                    best_epoch,
                    best_metric,
                )
                break
    train_time = time.perf_counter() - start_train

    if best_metrics is None or best_y_true is None or best_y_pred is None:
        raise RuntimeError("Training finished without a validation result.")
    metrics = best_metrics
    y_true = best_y_true
    y_pred = best_y_pred
    predict_time = best_predict_time
    LOGGER.info(
        "Validation completed in %.3f sec: accuracy=%.6f macro_f1=%.6f rows=%s",
        predict_time,
        metrics.accuracy,
        metrics.macro_f1,
        metrics.n_rows,
    )
    LOGGER.info("Per-class F1: %s", metrics.per_class_f1)
    LOGGER.info("Confusion matrix: %s", metrics.confusion_matrix)

    checkpoint_path = artifact_dir / BEST_CHECKPOINT_FILENAME
    preprocessor_path = _write_preprocessor(
        artifact_dir=artifact_dir,
        input_layout=model_metadata["input_layout"],
        train_config=training_config,
        model_name=model_name,
    )
    manifest_path_written = _write_service_manifest(
        artifact_dir=artifact_dir,
        model_name=model_name,
        model_version=f"local-seed{seed}-{artifact_dir.name}",
        input_layout=model_metadata["input_layout"],
        sample_rows=len(manifest),
        checkpoint_path=BEST_CHECKPOINT_FILENAME,
    )
    latest_manifest_path = _write_service_manifest(
        artifact_dir=service_artifact_dir,
        model_name=model_name,
        model_version=f"local-seed{seed}-{artifact_dir.name}",
        input_layout=model_metadata["input_layout"],
        sample_rows=len(manifest),
        checkpoint_path=str(checkpoint_path.relative_to(service_artifact_dir)),
        preprocessor_path=str(preprocessor_path.relative_to(service_artifact_dir)),
    )

    row = {
        "status": "completed",
        "experiment_id": f"{model_name}_{sample_size or 'full'}_seed{seed}",
        "model_name": model_metadata["name"],
        "model_family": model_metadata["family"],
        "training_mode": model_metadata["training_mode"],
        "pretrained": model_metadata["pretrained"],
        "manifest_path": str(manifest_path),
        "artifact_dir": str(artifact_dir),
        "service_artifact_dir": str(service_artifact_dir),
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
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "early_stopped": len(history) < epochs,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "batch_size": batch_size,
        "batch_size_auto": batch_size_auto,
        "batch_size_report": batch_size_report,
        "target_gpu_memory_utilization": target_gpu_memory_utilization,
        "max_auto_batch_size": max_auto_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "model_params": model_params or {},
        "scheduler": scheduler_name,
        "resume_from": str(resume_from) if resume_from is not None else None,
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
        "checkpoint_path": str(checkpoint_path),
        "resume_checkpoint_path": str(artifact_dir / RESUME_CHECKPOINT_FILENAME),
        "tensorboard_dir": str(tensorboard_dir),
        "preprocessor_path": str(preprocessor_path),
        "manifest_written_path": str(manifest_path_written),
        "latest_manifest_path": str(latest_manifest_path),
        "history": history,
    }
    _write_train_summary(
        artifact_dir=artifact_dir,
        status="completed",
        row=row,
    )
    append_experiment_result(output_path, row)
    LOGGER.info("Experiment result saved: %s", output_path)
    return row


def _dry_run_experiment(
    model: nn.Module,
    model_metadata: dict[str, str],
    model_params: dict[str, Any] | None,
    train_dataset: PartialDischargeDataset,
    valid_dataset: PartialDischargeDataset,
    manifest_path: Path,
    output_path: Path,
    split,
    split_name: str,
    sample_size: int | None,
    seed: int,
    epochs: int,
    run_device: torch.device,
    artifact_dir: Path,
    mixed_precision: str,
) -> dict[str, Any]:
    model.to(run_device)
    model_input_shape = _infer_input_shape(train_dataset=train_dataset, valid_dataset=valid_dataset)

    checkpoint_path = _write_dry_run_artifacts(
        artifact_dir=artifact_dir,
        model_name=str(model.name),
        input_layout=model_metadata["input_layout"],
        model_params=model_params,
    )
    preprocessor_path = _write_preprocessor(
        artifact_dir=artifact_dir,
        input_layout=model_metadata["input_layout"],
        train_config={
            "sample_size": sample_size or "full",
            "seed": seed,
            "epochs": epochs,
            "mixed_precision": mixed_precision,
            "model_name": str(model.name),
        },
        model_name=str(model.name),
    )
    manifest_path_written = _write_service_manifest(
        artifact_dir=artifact_dir,
        model_name=str(model.name),
        model_version=f"local-seed{seed}-dry-run",
        input_layout=model_metadata["input_layout"],
        sample_rows=len(train_dataset) + len(valid_dataset),
        ready=False,
    )

    row = {
        "status": "dry_run_ready",
        "experiment_id": f"{model.name}_{sample_size or 'full'}_seed{seed}",
        "model_name": model_metadata["name"],
        "model_family": model_metadata["family"],
        "training_mode": model_metadata["training_mode"],
        "pretrained": model_metadata["pretrained"],
        "manifest_path": str(manifest_path),
        "artifact_dir": str(artifact_dir),
        "split_type": split_name,
        "sample_size": sample_size or len(train_dataset) + len(valid_dataset),
        "train_rows": len(train_dataset),
        "valid_rows": len(valid_dataset),
        "valid_ratio": 1.0,
        "split_seed": seed,
        "epochs": 0,
        "input_layout": model_metadata["input_layout"],
        "device": str(run_device),
        "model_input_shape": list(model_input_shape),
        "checkpoint_path": str(checkpoint_path),
        "preprocessor_path": str(preprocessor_path),
        "manifest_written_path": str(manifest_path_written),
        "next_command": _real_training_command(
            model=model.name,
            manifest_path=manifest_path,
            sample_size=sample_size,
            epochs=epochs,
        ),
    }
    _write_train_summary(artifact_dir, status="dry_run_ready", row=row)
    append_experiment_result(output_path, row)
    return row


def _infer_input_shape(
    train_dataset: PartialDischargeDataset,
    valid_dataset: PartialDischargeDataset,
) -> tuple[int, ...]:
    if len(train_dataset) > 0:
        return tuple(train_dataset[0][0].shape)
    if len(valid_dataset) > 0:
        return tuple(valid_dataset[0][0].shape)
    raise ValueError("Both train and valid splits are empty. Increase sample size or adjust split settings.")


def _write_smoke_report(
    artifact_dir: Path,
    manifest_path: Path,
    output_path: Path,
    split,
    model_name: str,
    model_metadata: dict[str, str],
    model_params: dict[str, Any] | None,
    sample_size: int | None,
    seed: int,
    epochs: int,
    dry_run: bool,
) -> None:
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        LOGGER.info(
            "Dry run checks: model=%s layout=%s classes=%s split=%s sample_size=%s",
            model_name,
            model_metadata["input_layout"],
            list(model_metadata.keys()),
            split.split_type,
            sample_size or len(split.train) + len(split.valid),
        )
        return
    if len(split.train) == 0:
        raise ValueError("Training split is empty. Verify manifest split or sample settings.")
    if len(split.valid) == 0:
        raise ValueError("Validation split is empty. Verify manifest split or sample settings.")


def _write_checkpoint(
    artifact_dir: Path,
    model: nn.Module,
    model_name: str,
    input_layout: str,
    model_params: dict[str, Any] | None,
    filename: str = BEST_CHECKPOINT_FILENAME,
    epoch: int | None = None,
    metric_name: str | None = None,
    metric_value: float | None = None,
) -> Path:
    checkpoint_path = artifact_dir / filename
    payload: dict[str, Any] = _model_checkpoint_payload(model, model_name, input_layout, model_params)
    payload.update({"epoch": epoch, "metric_name": metric_name, "metric_value": metric_value})
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _write_resume_checkpoint(
    artifact_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    input_layout: str,
    model_params: dict[str, Any] | None,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> Path:
    checkpoint_path = artifact_dir / RESUME_CHECKPOINT_FILENAME
    payload: dict[str, Any] = _model_checkpoint_payload(model, model_name, input_layout, model_params)
    payload.update(
        {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        }
    )
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {"lr": learning_rate, "weight_decay": weight_decay}
    if torch.cuda.is_available() and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames:
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    learning_rate: float,
    epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if scheduler_name == "none":
        return None
    if scheduler_name != DEFAULT_SCHEDULER:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}. Supported: {DEFAULT_SCHEDULER}|none")
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=max(1, steps_per_epoch),
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=100.0,
    )


def _load_resume_state(
    resume_from: Path | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> dict[str, Any]:
    if resume_from is None:
        return {}
    if not resume_from.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_from}")
    checkpoint = torch.load(resume_from, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("resume checkpoint must be a serialized dictionary.")
    state = checkpoint.get("model_state_dict")
    if isinstance(state, dict):
        model.load_state_dict(state)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and isinstance(scheduler_state, dict):
        scheduler.load_state_dict(scheduler_state)
    LOGGER.info(
        "Resumed time-series training from %s at epoch=%s best_epoch=%s best_metric=%s",
        resume_from,
        checkpoint.get("epoch"),
        checkpoint.get("best_epoch"),
        checkpoint.get("best_metric"),
    )
    return checkpoint


def _model_checkpoint_payload(
    model: nn.Module,
    model_name: str,
    input_layout: str,
    model_params: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "input_layout": input_layout,
        "model_params": model_params or {},
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
    }


def _write_preprocessor(
    artifact_dir: Path,
    input_layout: str,
    train_config: dict[str, Any],
    model_name: str,
) -> Path:
    preprocessor_path = artifact_dir / PREPROCESSOR_FILENAME
    payload = {
        "model_name": model_name,
        "input_layout": input_layout,
        "normalize": True,
        "training_config": train_config,
        "label_map": {str(key): value for key, value in LABEL_ID_TO_NAME.items()},
    }
    preprocessor_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return preprocessor_path


def _write_service_manifest(
    artifact_dir: Path,
    model_name: str,
    model_version: str,
    input_layout: str,
    sample_rows: int,
    ready: bool = True,
    checkpoint_path: str = BEST_CHECKPOINT_FILENAME,
    preprocessor_path: str = PREPROCESSOR_FILENAME,
) -> Path:
    manifest_payload = {
        "task": "time_series",
        "model_name": model_name,
        "model_version": model_version,
        "framework": "pytorch",
        "entrypoint": "ml.timeseries.src.service_adapter:load_adapter",
        "checkpoint_path": checkpoint_path,
        "preprocessor_path": preprocessor_path,
        "label_map": {str(key): value for key, value in LABEL_ID_TO_NAME.items()},
        "input_spec": {
            "modality": "time_series_csv",
            "schema_version": "1.0",
            "shape": [20, 7680] if input_layout == "channel_first" else [7680, 20],
            "dtype": "float32",
            "notes": "CSV artifact path is passed through TimeSeriesToolInput.csv_path.",
        },
        "output_spec": {
            "schema_version": "1.0",
            "required_fields": ["label_id", "confidence", "probabilities", "features"],
            "notes": "The backend output is normalized to TimeSeriesResult.",
        },
        "thresholds": {
            "min_confidence": 0.72,
            "review_confidence": 0.55,
        },
        "runtime": {
            "device": "auto",
            "batch_size": 1,
            "artifact_rows": sample_rows,
            "ready": ready,
        },
    }
    manifest_path = artifact_dir / MODEL_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _timestamped_run_dir(base_dir: Path, model_name: str) -> Path:
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    return base_dir / safe_model_name / started_at


def _write_train_summary(
    artifact_dir: Path,
    status: str,
    row: dict[str, Any],
) -> Path:
    payload = dict(row)
    payload["status"] = status
    summary_path = artifact_dir / TRAIN_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def _write_dry_run_artifacts(
    artifact_dir: Path,
    model_name: str,
    input_layout: str,
    model_params: dict[str, Any] | None,
) -> Path:
    payload = {
        "status": "dry_run_ready",
        "model_name": model_name,
        "input_layout": input_layout,
        "model_params": model_params or {},
    }
    dry_run_marker = artifact_dir / "dry_run_summary.json"
    dry_run_marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dry_run_marker


def _real_training_command(
    model: str,
    manifest_path: Path,
    sample_size: int | None,
    epochs: int,
) -> str:
    return (
        "python ml/timeseries/train.py "
        f"--model {model} "
        f"--manifest {manifest_path} "
        f"--epochs {epochs} "
        f"{f'--sample-size {sample_size} ' if sample_size is not None else ''}"
    ).strip()
