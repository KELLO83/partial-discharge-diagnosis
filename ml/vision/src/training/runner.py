from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ml.vision.src.data.dataset import PrpdImageDataset
from ml.vision.src.data.manifest import load_vision_manifest, split_vision_manifest
from ml.vision.src.eval import classification_metrics
from ml.vision.src.models import SmallPrpdCnn
from ml.vision.src.schema import DEFAULT_NUM_CLASSES, PD_LABELS_KO, VisionManifestSplit, VisionTrainingConfig
from ml.training.artifacts import relative_artifact_path, timestamped_model_dir
from ml.timeseries.src.torch_runtime import autocast_dtype, autocast_enabled

LOGGER = logging.getLogger(__name__)
CHECKPOINT_FILENAME = "checkpoint.pt"
BEST_CHECKPOINT_FILENAME = "best.pt"
RESUME_CHECKPOINT_FILENAME = "resumet.pt"
SUMMARY_FILENAME = "train_summary.json"
CONFIG_FILENAME = "training_config.json"
LABEL_MAPPING_FILENAME = "label_mapping.json"
SERVICE_MANIFEST_FILENAME = "model_manifest.json"
PREPROCESSOR_FILENAME = "preprocessor.json"
EVIDENCE_CONTEXT_FILENAME = "evidence_context.csv"
TENSORBOARD_DIRNAME = "tensorboard"


@dataclass(frozen=True, slots=True)
class VisionTrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    device: torch.device


def run_vision_training(config: VisionTrainingConfig) -> Path:
    _set_seed(config.seed)
    split = _prepare_split(config)
    if not config.dry_run and config.publish_dir is None:
        config = replace(
            config,
            publish_dir=config.output_dir,
            output_dir=timestamped_model_dir(config.output_dir, config.model_name),
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = config.output_dir / PREPROCESSOR_FILENAME
    _write_json(config.output_dir / CONFIG_FILENAME, _json_ready_config(config))
    _write_json(config.output_dir / LABEL_MAPPING_FILENAME, _label_mapping())
    _write_preprocessor(preprocessor_path, config)
    if config.dry_run:
        _write_service_manifest(config, ready=False)
        return _write_dry_run_summary(config, split, preprocessor_path)
    _assert_trainable_split(split)
    return _train_and_save(config, split)


def _prepare_split(config: VisionTrainingConfig) -> VisionManifestSplit:
    manifest = load_vision_manifest(config.manifest_path)
    split = split_vision_manifest(manifest, config)
    _raise_for_missing_images(split)
    return split


def _write_dry_run_summary(
    config: VisionTrainingConfig, split: VisionManifestSplit, preprocessor_path: Path
) -> Path:
    if split.train_rows.empty or split.valid_rows.empty:
        raise ValueError("Vision dry run requires both train and valid splits. Adjust sample-size or valid-ratio.")
    summary_path = config.output_dir / SUMMARY_FILENAME
    _write_json(
        summary_path,
        {
            "status": "dry_run_ready",
            "model_name": config.model_name,
            "train_rows": len(split.train_rows),
            "valid_rows": len(split.valid_rows),
            "split_type": split.split_type,
            "image_size": config.image_size,
            "preprocessor_path": str(preprocessor_path),
            "manifest_path": str(config.output_dir / SERVICE_MANIFEST_FILENAME),
            "next_command": _real_training_command(config),
        },
    )
    return summary_path


def _train_and_save(config: VisionTrainingConfig, split: VisionManifestSplit) -> Path:
    state = _build_train_state(config)
    batch_size, batch_report = _resolve_batch_size(config, state, split.train_rows)
    effective_config = replace(config, batch_size=batch_size)
    train_loader = _train_data_loader(split.train_rows, effective_config, batch_size)
    valid_loader = _valid_data_loader(split.valid_rows, effective_config, batch_size)
    scheduler = _build_scheduler(state.optimizer, effective_config, len(train_loader))
    resume_state = _load_resume_state(state, effective_config.resume_from, scheduler)
    start_epoch = int(resume_state.get("epoch", 0))

    history = []
    best_loss = float(resume_state.get("best_loss", float("inf")))
    best_epoch = int(resume_state.get("best_epoch", start_epoch))
    epochs_without_improvement = 0
    best_valid_output: dict[str, Any] | None = None
    tensorboard_dir = effective_config.output_dir / TENSORBOARD_DIRNAME
    with SummaryWriter(log_dir=str(tensorboard_dir)) as writer:
        for epoch in range(start_epoch + 1, effective_config.epochs + 1):
            train_loss = _train_epoch(state, train_loader, effective_config, scheduler)
            valid_output = _predict(state.model, valid_loader, state.device, state.criterion, effective_config)
            metrics = classification_metrics(valid_output["targets"], valid_output["predictions"], DEFAULT_NUM_CLASSES)
            valid_loss = float(valid_output["loss"])
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("eval/loss", valid_loss, epoch)
            writer.add_scalar("eval/accuracy", metrics["accuracy"], epoch)
            if scheduler is not None:
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
            improved = valid_loss < best_loss - effective_config.min_delta
            if improved:
                best_loss = valid_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_valid_output = valid_output
                _save_checkpoint(state, effective_config, BEST_CHECKPOINT_FILENAME, epoch=epoch, valid_loss=valid_loss)
            else:
                epochs_without_improvement += 1
            _save_resume_checkpoint(
                state,
                effective_config,
                scheduler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_loss=best_loss,
            )
            history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "valid_metrics": metrics})
            LOGGER.info(
                "vision epoch=%s train_loss=%.6f valid_loss=%.6f valid_accuracy=%.4f best_epoch=%s",
                epoch,
                train_loss,
                valid_loss,
                metrics["accuracy"],
                best_epoch,
            )
            if (
                effective_config.early_stop_patience > 0
                and epochs_without_improvement >= effective_config.early_stop_patience
            ):
                LOGGER.info("vision early stopping at epoch=%s patience=%s", epoch, effective_config.early_stop_patience)
                break

    checkpoint_path = effective_config.output_dir / BEST_CHECKPOINT_FILENAME
    final_valid_output = best_valid_output or valid_output
    evidence_path = _write_evidence_context(effective_config, final_valid_output)
    preprocessor_path = effective_config.output_dir / PREPROCESSOR_FILENAME
    _write_preprocessor(preprocessor_path, effective_config)
    manifest_path = _write_service_manifest(
        effective_config,
        ready=True,
        checkpoint_path=checkpoint_path,
        preprocessor_path=preprocessor_path,
    )
    summary_path = effective_config.output_dir / SUMMARY_FILENAME
    _write_json(
        summary_path,
        {
            "status": "trained",
            "model_name": effective_config.model_name,
            "output_dir": str(effective_config.output_dir),
            "publish_dir": str(effective_config.publish_dir) if effective_config.publish_dir is not None else None,
            "checkpoint_path": str(checkpoint_path),
            "resume_checkpoint_path": str(effective_config.output_dir / RESUME_CHECKPOINT_FILENAME),
            "tensorboard_dir": str(tensorboard_dir),
            "evidence_context_path": str(evidence_path),
            "train_rows": len(split.train_rows),
            "valid_rows": len(split.valid_rows),
            "split_type": split.split_type,
            "manifest_path": str(manifest_path),
            "preprocessor_path": str(preprocessor_path),
            "batch_size": batch_size,
            "batch_size_report": batch_report,
            "best_epoch": best_epoch,
            "best_valid_loss": best_loss,
            "history": history,
            "final_metrics": history[-1]["valid_metrics"],
        },
    )
    return summary_path


def _build_train_state(config: VisionTrainingConfig) -> VisionTrainState:
    device = _resolve_device(config.device)
    _configure_cuda_memory(device, config.gpu_memory_fraction)
    model = SmallPrpdCnn(num_classes=DEFAULT_NUM_CLASSES).to(device)
    return VisionTrainState(
        model=model,
        optimizer=_build_optimizer(model, config),
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )


def _train_data_loader(frame: pd.DataFrame, config: VisionTrainingConfig, batch_size: int) -> DataLoader:
    dataset = PrpdImageDataset(frame, image_size=config.image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device in {"auto", "cuda"} and torch.cuda.is_available(),
    )


def _valid_data_loader(frame: pd.DataFrame, config: VisionTrainingConfig, batch_size: int) -> DataLoader:
    dataset = PrpdImageDataset(frame, image_size=config.image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device in {"auto", "cuda"} and torch.cuda.is_available(),
    )


def _train_epoch(
    state: VisionTrainState,
    loader: DataLoader,
    config: VisionTrainingConfig,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> float:
    state.model.train()
    total_loss = 0.0
    total_rows = 0
    for images, labels, _ in tqdm(loader, desc="vision-train", leave=False):
        images = images.to(state.device, non_blocking=True)
        labels = labels.to(state.device, non_blocking=True)
        state.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=state.device.type,
            dtype=autocast_dtype(config.mixed_precision),
            enabled=autocast_enabled(config.mixed_precision, state.device),
        ):
            loss = state.criterion(state.model(images), labels)
        loss.backward()
        state.optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item()) * int(labels.shape[0])
        total_rows += int(labels.shape[0])
    return total_loss / max(1, total_rows)


def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    config: VisionTrainingConfig,
) -> dict[str, Any]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    sample_ids: list[str] = []
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for images, labels, ids in tqdm(loader, desc="vision-valid", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=autocast_dtype(config.mixed_precision),
                enabled=autocast_enabled(config.mixed_precision, device),
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            batch_probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            total_loss += float(loss.item()) * int(labels.shape[0])
            total_rows += int(labels.shape[0])
            probabilities.append(batch_probabilities)
            predictions.append(batch_probabilities.argmax(axis=1))
            targets.append(labels.cpu().numpy())
            sample_ids.extend(str(sample_id) for sample_id in ids)
    return {
        "sample_ids": sample_ids,
        "targets": np.concatenate(targets),
        "predictions": np.concatenate(predictions),
        "probabilities": np.concatenate(probabilities),
        "loss": total_loss / max(1, total_rows),
    }


def _save_checkpoint(
    state: VisionTrainState,
    config: VisionTrainingConfig,
    filename: str,
    *,
    epoch: int,
    valid_loss: float,
) -> Path:
    checkpoint_path = config.output_dir / filename
    torch.save(
        {
            "model_state_dict": {key: value.cpu() for key, value in state.model.state_dict().items()},
            "model_name": config.model_name,
            "model_class": "SmallPrpdCnn",
            "image_size": config.image_size,
            "label_mapping": _label_mapping(),
            "epoch": epoch,
            "valid_loss": valid_loss,
        },
        checkpoint_path,
    )
    return checkpoint_path


def _save_resume_checkpoint(
    state: VisionTrainState,
    config: VisionTrainingConfig,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    *,
    epoch: int,
    best_epoch: int,
    best_loss: float,
) -> Path:
    checkpoint_path = config.output_dir / RESUME_CHECKPOINT_FILENAME
    torch.save(
        {
            "model_state_dict": {key: value.cpu() for key, value in state.model.state_dict().items()},
            "optimizer_state_dict": state.optimizer.state_dict(),
            "model_name": config.model_name,
            "model_class": "SmallPrpdCnn",
            "image_size": config.image_size,
            "label_mapping": _label_mapping(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        checkpoint_path,
    )
    return checkpoint_path


def _write_evidence_context(config: VisionTrainingConfig, output: dict[str, Any]) -> Path:
    rows = []
    for index, sample_id in enumerate(output["sample_ids"]):
        probabilities = output["probabilities"][index]
        predicted_label = int(output["predictions"][index])
        row = {
            "sample_id": sample_id,
            "vision_model_name": config.model_name,
            "vision_pred_label_id": predicted_label,
            "vision_confidence": float(probabilities[predicted_label]),
        }
        row.update({f"vision_prob_{label_id}": float(probabilities[label_id]) for label_id in range(DEFAULT_NUM_CLASSES)})
        rows.append(row)
    evidence_path = config.output_dir / EVIDENCE_CONTEXT_FILENAME
    pd.DataFrame(rows).to_csv(evidence_path, index=False)
    return evidence_path


def _write_service_manifest(
    config: VisionTrainingConfig,
    ready: bool = True,
    checkpoint_path: Path | None = None,
    preprocessor_path: Path | None = None,
) -> Path:
    manifest_dir = config.publish_dir or config.output_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / SERVICE_MANIFEST_FILENAME
    checkpoint = checkpoint_path or (config.output_dir / BEST_CHECKPOINT_FILENAME)
    preprocessor = preprocessor_path or (config.output_dir / PREPROCESSOR_FILENAME)
    _write_json(
        manifest_path,
        {
            "task": "vision",
            "model_name": config.model_name,
            "model_version": f"local-seed-{config.seed}",
            "framework": "pytorch",
            "entrypoint": "ml.vision.src.service_adapter:load_adapter",
            "checkpoint_path": relative_artifact_path(manifest_dir, checkpoint),
            "preprocessor_path": relative_artifact_path(manifest_dir, preprocessor),
            "label_map": {str(key): value for key, value in PD_LABELS_KO.items()},
            "input_spec": {
                "modality": "prpd_image",
                "schema_version": "1.0",
                "shape": [3, config.image_size, config.image_size],
                "dtype": "float32",
                "notes": "RGB PRPD PNG resized and ImageNet-normalized.",
            },
            "output_spec": {
                "schema_version": "1.0",
                "required_fields": ["label_id", "confidence", "probabilities", "evidence"],
            },
            "runtime": {"device": config.device, "image_size": config.image_size, "ready": ready},
        },
    )
    return manifest_path


def _write_preprocessor(preprocessor_path: Path, config: VisionTrainingConfig) -> Path:
    payload = {
        "model_name": config.model_name,
        "image_size": config.image_size,
        "seed": config.seed,
        "device": config.device,
        "num_workers": config.num_workers,
        "normalize": True,
        "train_rows": len(load_vision_manifest(config.manifest_path)),
        "label_map": {str(key): value for key, value in PD_LABELS_KO.items()},
    }
    preprocessor_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return preprocessor_path


def _assert_trainable_split(split: VisionManifestSplit) -> None:
    if split.train_rows.empty:
        raise ValueError("Vision training split is empty.")
    if split.valid_rows.empty:
        raise ValueError("Vision validation split is empty. Increase --sample-size or lower --valid-ratio.")


def _raise_for_missing_images(split: VisionManifestSplit) -> None:
    combined = pd.concat([split.train_rows, split.valid_rows], ignore_index=True)
    missing_paths = [path for path in combined["resolved_image_path"].astype(str) if not Path(path).exists()]
    if missing_paths:
        examples = ", ".join(missing_paths[:3])
        raise FileNotFoundError(f"Vision manifest references missing images: {examples}")


def _load_resume_state(
    state: VisionTrainState,
    resume_from: Path | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> dict[str, Any]:
    if resume_from is None:
        return {}
    checkpoint = torch.load(resume_from, map_location=state.device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Vision resume checkpoint must be a serialized dictionary.")
    state.model.load_state_dict(checkpoint["model_state_dict"])
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if isinstance(optimizer_state, dict):
        state.optimizer.load_state_dict(optimizer_state)
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and isinstance(scheduler_state, dict):
        scheduler.load_state_dict(scheduler_state)
    start_epoch = int(checkpoint.get("epoch", 0))
    LOGGER.info("vision resume loaded: %s epoch=%s", resume_from, start_epoch)
    return checkpoint


def _resolve_batch_size(
    config: VisionTrainingConfig,
    state: VisionTrainState,
    train_rows: pd.DataFrame,
) -> tuple[int, dict[str, Any]]:
    if config.batch_size is not None:
        return config.batch_size, {"mode": "manual", "batch_size": config.batch_size}
    if state.device.type != "cuda":
        return min(16, max(1, len(train_rows))), {"mode": "cpu_default"}
    device_index = _cuda_device_index(state.device)
    props = torch.cuda.get_device_properties(device_index)
    target_bytes = int(props.total_memory * config.gpu_memory_fraction)
    start_batch = max(1, min(config.auto_batch_start_size, len(train_rows), config.max_auto_batch_size))
    probe_batch = start_batch
    while probe_batch >= 1:
        try:
            peak_reserved = _try_train_step(state, config.image_size, probe_batch, config)
            break
        except torch.cuda.OutOfMemoryError:
            _clear_cuda_state(state.device)
            probe_batch //= 2
    else:
        raise RuntimeError("Unable to run vision batch_size=1 on CUDA.")

    per_sample_bytes = max(1, peak_reserved // max(1, probe_batch))
    candidate = max(1, min(target_bytes // per_sample_bytes, len(train_rows), config.max_auto_batch_size))
    while candidate >= 1:
        try:
            peak_reserved = _try_train_step(state, config.image_size, int(candidate), config)
            if peak_reserved <= target_bytes or candidate == 1:
                break
            candidate //= 2
        except torch.cuda.OutOfMemoryError:
            _clear_cuda_state(state.device)
            candidate //= 2
    batch_size = max(1, int(candidate))
    report = {
        "mode": "auto",
        "gpu_memory_fraction": config.gpu_memory_fraction,
        "resolved_batch_size": batch_size,
        "peak_reserved_bytes": int(peak_reserved),
        "gpu_total_gb": round(float(props.total_memory) / (1024**3), 4),
        "estimated_peak_memory_utilization": round(float(peak_reserved) / max(1, float(props.total_memory)), 6),
        "max_auto_batch_size": config.max_auto_batch_size,
    }
    LOGGER.info(
        "vision auto batch resolved: batch_size=%s target=%.2f estimated_utilization=%.3f",
        batch_size,
        config.gpu_memory_fraction,
        report["estimated_peak_memory_utilization"],
    )
    return batch_size, report


def _try_train_step(
    state: VisionTrainState,
    image_size: int,
    batch_size: int,
    config: VisionTrainingConfig,
) -> int:
    _clear_cuda_state(state.device)
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)
    images = torch.randn((batch_size, 3, image_size, image_size), device=state.device)
    labels = torch.zeros(batch_size, dtype=torch.long, device=state.device)
    with torch.amp.autocast(
        device_type=state.device.type,
        dtype=autocast_dtype(config.mixed_precision),
        enabled=autocast_enabled(config.mixed_precision, state.device),
    ):
        loss = state.criterion(state.model(images), labels)
    loss.backward()
    state.optimizer.zero_grad(set_to_none=True)
    device_index = _cuda_device_index(state.device)
    torch.cuda.synchronize(device_index)
    peak_reserved = torch.cuda.max_memory_reserved(device_index)
    del images, labels, loss
    _clear_cuda_state(state.device)
    return int(peak_reserved)


def _clear_cuda_state(device: torch.device) -> None:
    if device.type == "cuda":
        device_index = _cuda_device_index(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_index)
        torch.cuda.synchronize(device_index)


def _cuda_device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def _configure_cuda_memory(device: torch.device, gpu_memory_fraction: float) -> None:
    if device.type != "cuda":
        return
    if not 0 < gpu_memory_fraction <= 1:
        raise ValueError("gpu memory fraction must be in (0, 1].")
    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, _cuda_device_index(device))


def _build_optimizer(model: nn.Module, config: VisionTrainingConfig) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if torch.cuda.is_available() and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames:
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: VisionTrainingConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.scheduler == "none":
        return None
    if config.scheduler != "onecycle":
        raise ValueError(f"Unsupported vision scheduler: {config.scheduler}. Supported: onecycle|none")
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=max(1, steps_per_epoch),
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=100.0,
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for vision training, but CUDA is not available.")
    return torch.device(requested)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_ready_config(config: VisionTrainingConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["manifest_path"] = str(config.manifest_path)
    payload["output_dir"] = str(config.output_dir)
    payload["publish_dir"] = str(config.publish_dir) if config.publish_dir is not None else None
    payload["resume_from"] = str(config.resume_from) if config.resume_from is not None else None
    return payload


def _label_mapping() -> dict[str, str]:
    return {str(key): value for key, value in PD_LABELS_KO.items()}


def _real_training_command(config: VisionTrainingConfig) -> str:
    batch_arg = f"--batch-size {config.batch_size} " if config.batch_size is not None else ""
    return (
        "python ml/vision/train.py "
        f"--manifest {config.manifest_path} "
        f"--output-dir {config.publish_dir or config.output_dir} "
        f"--epochs {config.epochs} "
        f"{batch_arg}"
        f"--gpu-memory-fraction {config.gpu_memory_fraction}"
    )
