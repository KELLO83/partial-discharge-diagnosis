from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ml.vision.src.data.dataset import PrpdImageDataset
from ml.vision.src.data.manifest import load_vision_manifest, split_vision_manifest
from ml.vision.src.eval import classification_metrics
from ml.vision.src.models import SmallPrpdCnn
from ml.vision.src.schema import DEFAULT_NUM_CLASSES, PD_LABELS_KO, VisionManifestSplit, VisionTrainingConfig

LOGGER = logging.getLogger(__name__)
CHECKPOINT_FILENAME = "checkpoint.pt"
SUMMARY_FILENAME = "train_summary.json"
CONFIG_FILENAME = "training_config.json"
LABEL_MAPPING_FILENAME = "label_mapping.json"
SERVICE_MANIFEST_FILENAME = "model_manifest.json"
EVIDENCE_CONTEXT_FILENAME = "evidence_context.csv"


@dataclass(frozen=True, slots=True)
class VisionTrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    device: torch.device


def run_vision_training(config: VisionTrainingConfig) -> Path:
    _set_seed(config.seed)
    split = _prepare_split(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / CONFIG_FILENAME, _json_ready_config(config))
    _write_json(config.output_dir / LABEL_MAPPING_FILENAME, _label_mapping())
    if config.dry_run:
        return _write_dry_run_summary(config, split)
    _assert_trainable_split(split)
    return _train_and_save(config, split)


def _prepare_split(config: VisionTrainingConfig) -> VisionManifestSplit:
    manifest = load_vision_manifest(config.manifest_path)
    split = split_vision_manifest(manifest, config)
    _raise_for_missing_images(split)
    return split


def _write_dry_run_summary(config: VisionTrainingConfig, split: VisionManifestSplit) -> Path:
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
            "next_command": _real_training_command(config),
        },
    )
    return summary_path


def _train_and_save(config: VisionTrainingConfig, split: VisionManifestSplit) -> Path:
    state = _build_train_state(config)
    train_loader = _train_data_loader(split.train_rows, config)
    valid_loader = _valid_data_loader(split.valid_rows, config)

    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(state, train_loader)
        valid_output = _predict(state.model, valid_loader, state.device)
        metrics = classification_metrics(valid_output["targets"], valid_output["predictions"], DEFAULT_NUM_CLASSES)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_metrics": metrics})
        LOGGER.info("vision epoch=%s train_loss=%.6f valid_accuracy=%.4f", epoch, train_loss, metrics["accuracy"])

    checkpoint_path = _save_checkpoint(state.model, config)
    evidence_path = _write_evidence_context(config, valid_output)
    _write_service_manifest(config)
    summary_path = config.output_dir / SUMMARY_FILENAME
    _write_json(
        summary_path,
        {
            "status": "trained",
            "model_name": config.model_name,
            "checkpoint_path": str(checkpoint_path),
            "evidence_context_path": str(evidence_path),
            "train_rows": len(split.train_rows),
            "valid_rows": len(split.valid_rows),
            "split_type": split.split_type,
            "history": history,
            "final_metrics": history[-1]["valid_metrics"],
        },
    )
    return summary_path


def _build_train_state(config: VisionTrainingConfig) -> VisionTrainState:
    device = _resolve_device(config.device)
    model = SmallPrpdCnn(num_classes=DEFAULT_NUM_CLASSES).to(device)
    return VisionTrainState(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=config.learning_rate),
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )


def _train_data_loader(frame: pd.DataFrame, config: VisionTrainingConfig) -> DataLoader:
    dataset = PrpdImageDataset(frame, image_size=config.image_size)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device in {"auto", "cuda"} and torch.cuda.is_available(),
    )


def _valid_data_loader(frame: pd.DataFrame, config: VisionTrainingConfig) -> DataLoader:
    dataset = PrpdImageDataset(frame, image_size=config.image_size)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device in {"auto", "cuda"} and torch.cuda.is_available(),
    )


def _train_epoch(state: VisionTrainState, loader: DataLoader) -> float:
    state.model.train()
    total_loss = 0.0
    total_rows = 0
    for images, labels, _ in tqdm(loader, desc="vision-train", leave=False):
        images = images.to(state.device, non_blocking=True)
        labels = labels.to(state.device, non_blocking=True)
        state.optimizer.zero_grad(set_to_none=True)
        loss = state.criterion(state.model(images), labels)
        loss.backward()
        state.optimizer.step()
        total_loss += float(loss.item()) * int(labels.shape[0])
        total_rows += int(labels.shape[0])
    return total_loss / max(1, total_rows)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for images, labels, ids in tqdm(loader, desc="vision-valid", leave=False):
            logits = model(images.to(device, non_blocking=True))
            batch_probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            probabilities.append(batch_probabilities)
            predictions.append(batch_probabilities.argmax(axis=1))
            targets.append(labels.cpu().numpy())
            sample_ids.extend(str(sample_id) for sample_id in ids)
    return {
        "sample_ids": sample_ids,
        "targets": np.concatenate(targets),
        "predictions": np.concatenate(predictions),
        "probabilities": np.concatenate(probabilities),
    }


def _save_checkpoint(model: nn.Module, config: VisionTrainingConfig) -> Path:
    checkpoint_path = config.output_dir / CHECKPOINT_FILENAME
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": config.model_name,
            "model_class": "SmallPrpdCnn",
            "image_size": config.image_size,
            "label_mapping": _label_mapping(),
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


def _write_service_manifest(config: VisionTrainingConfig) -> Path:
    manifest_path = config.output_dir / SERVICE_MANIFEST_FILENAME
    _write_json(
        manifest_path,
        {
            "task": "vision",
            "model_name": config.model_name,
            "model_version": f"local-seed-{config.seed}",
            "framework": "pytorch",
            "entrypoint": "ml.vision.src.service_adapter:load_adapter",
            "checkpoint_path": CHECKPOINT_FILENAME,
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
            "runtime": {"device": config.device, "image_size": config.image_size},
        },
    )
    return manifest_path


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
    return payload


def _label_mapping() -> dict[str, str]:
    return {str(key): value for key, value in PD_LABELS_KO.items()}


def _real_training_command(config: VisionTrainingConfig) -> str:
    return (
        "python ml/vision/train.py "
        f"--manifest {config.manifest_path} "
        f"--output-dir {config.output_dir} "
        f"--epochs {config.epochs} "
        f"--batch-size {config.batch_size}"
    )
