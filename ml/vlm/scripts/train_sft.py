from __future__ import annotations

import argparse
import importlib
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.timeseries.src.torch_runtime import (
    CompileConfig,
    SdpaProbeConfig,
    autocast_dtype,
    log_sdpa_backend_report,
    maybe_compile_model,
)
from ml.training.artifacts import relative_artifact_path

LOGGER = logging.getLogger(__name__)
TRAINING_CONFIG_FILENAME = "training_config.json"
PREPROCESSOR_FILENAME = "preprocessor.json"
PROCESSOR_DIRNAME = "processor"
CHECKPOINT_DIRNAME = "best.pt"
RESUME_CHECKPOINT_DIRNAME = "resumet.pt"
MODEL_MANIFEST_FILENAME = "model_manifest.json"
SUMMARY_FILENAME = "train_summary.json"
DRY_RUN_SUMMARY_FILENAME = "dry_run_summary.json"
SERVICE_ARTIFACT_TASK = "vlm"
DEFAULT_MAX_NEW_TOKENS = 512
TENSORBOARD_DIRNAME = "tensorboard"
TORCH_SDPA_ATTENTION = "sdpa"
SUPPORTED_ATTENTION_IMPLEMENTATIONS = (TORCH_SDPA_ATTENTION, "eager")

class TrainingRiskError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model_id: str
    dataset: str
    output_dir: str
    publish_dir: str | None
    load_in_4bit: bool
    full_finetune: bool
    max_steps: int
    save_steps: int = 1
    early_stop_patience: int = 3
    min_delta: float = 0.0
    resume_from: str | None = None
    gpu_memory_fraction: float = 0.9
    eval_ratio: float = 0.2
    weight_decay: float = 0.0
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    training_profile: str = "manual"
    training_strategy: str = "4bit_qlora_sft_text_projector_only"
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    train_vision_tower: bool = False
    train_projector: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    target_modules: str = "all-linear"
    precision: str = "fp16"
    image_max_pixels: str = "512x512"
    flash_attention: bool = False
    attn_implementation: str = "sdpa"
    torch_compile: bool = False
    torch_compile_mode: str = "default"


def build_training_config(
    model_id: str,
    dataset: str,
    output_dir: str,
    load_in_4bit: bool,
    full_finetune: bool,
    risk_override: bool,
    max_steps: int,
    publish_dir: str | None = None,
    save_steps: int = 1,
    early_stop_patience: int = 3,
    min_delta: float = 0.0,
    resume_from: str | None = None,
    gpu_memory_fraction: float = 0.9,
    eval_ratio: float = 0.2,
    weight_decay: float = 0.0,
    optim: str = "paged_adamw_8bit",
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.03,
    attn_implementation: str = "sdpa",
    precision: str = "fp16",
    torch_compile: bool = False,
    torch_compile_mode: str = "default",
) -> TrainingConfig:
    if full_finetune and not risk_override:
        raise TrainingRiskError("Full fine-tuning is blocked for RTX 4060 Laptop 8GB without explicit override.")
    if not load_in_4bit and not risk_override:
        raise TrainingRiskError("VLM SFT on 8GB must use --load-in-4bit unless --i-understand-8gb-risk is set.")
    if max_steps < 1:
        raise ValueError("--max-steps must be at least 1.")
    if save_steps < 1:
        raise ValueError("--save-steps must be at least 1.")
    if not 0 < gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0, 1].")
    if not 0 <= eval_ratio < 1:
        raise ValueError("--eval-ratio must be in [0, 1).")
    if attn_implementation not in SUPPORTED_ATTENTION_IMPLEMENTATIONS:
        supported = "|".join(SUPPORTED_ATTENTION_IMPLEMENTATIONS)
        raise ValueError(
            f"Unsupported attention implementation: {attn_implementation}. "
            f"Use PyTorch SDPA/eager only: {supported}."
        )
    if torch_compile and load_in_4bit and not risk_override:
        raise TrainingRiskError("torch.compile with 4-bit VLM loading is blocked without --i-understand-8gb-risk.")
    return TrainingConfig(
        model_id=model_id,
        dataset=dataset,
        output_dir=output_dir,
        publish_dir=publish_dir,
        load_in_4bit=load_in_4bit,
        full_finetune=full_finetune,
        max_steps=max_steps,
        save_steps=save_steps,
        early_stop_patience=early_stop_patience,
        min_delta=min_delta,
        resume_from=resume_from,
        gpu_memory_fraction=gpu_memory_fraction,
        eval_ratio=eval_ratio,
        weight_decay=weight_decay,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        attn_implementation=attn_implementation,
        precision=precision,
        torch_compile=torch_compile,
        torch_compile_mode=torch_compile_mode,
    )


def write_dry_run_artifacts(config: TrainingConfig) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / TRAINING_CONFIG_FILENAME
    preprocessor_path = output_dir / PREPROCESSOR_FILENAME
    checkpoint_dir = output_dir / CHECKPOINT_DIRNAME
    manifest_path = _write_service_manifest(output_dir, config, ready=False, checkpoint=checkpoint_dir)
    if config.publish_dir is not None:
        _write_service_manifest(Path(config.publish_dir), config, ready=False, checkpoint=checkpoint_dir, preprocessor=preprocessor_path)
    config_payload = asdict(config)
    train_rows = _count_jsonl_rows(Path(config.dataset))
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_preprocessor(preprocessor_path, config)
    summary_path = output_dir / DRY_RUN_SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(
            {
                "status": "dry_run_ready",
                "config": config_payload,
                "train_rows": train_rows,
                "valid_rows": train_rows,
                "adapter_saved": False,
                "manifest_path": str(manifest_path),
                "preprocessor_path": str(preprocessor_path),
                "checkpoint_path": str(checkpoint_dir),
                "tensorboard_dir": str(output_dir / TENSORBOARD_DIRNAME),
                "peak_vram_mb": None,
                "next_command": _real_training_command(config),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


def run_sft_training(config: TrainingConfig) -> Path:
    torch = _load_module("torch")
    datasets = _load_module("datasets")
    peft = _load_module("peft")
    transformers = _load_module("transformers")
    trl = _load_module("trl")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / TRAINING_CONFIG_FILENAME
    preprocessor_path = output_dir / PREPROCESSOR_FILENAME
    config_path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_preprocessor(preprocessor_path, config)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    sdpa_report = None
    if config.attn_implementation == "sdpa":
        sdpa_report = log_sdpa_backend_report(
            LOGGER,
            "VLM SFT",
            SdpaProbeConfig(
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=autocast_dtype(config.precision) or torch.float32,
            ),
        )
    processor = transformers.AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation=config.attn_implementation,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model, torch_compile_report = maybe_compile_model(
        model=model,
        config=CompileConfig(enabled=config.torch_compile, mode=config.torch_compile_mode),
        logger=LOGGER,
    )
    raw_dataset = datasets.load_dataset("json", data_files=config.dataset, split="train")
    train_dataset, eval_dataset = _split_train_eval_dataset(raw_dataset, config.eval_ratio)
    peft_config = None
    if not config.full_finetune:
        peft_config = peft.LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            task_type="CAUSAL_LM",
        )
    processor_dir = output_dir / PROCESSOR_DIRNAME
    best_checkpoint_dir = output_dir / CHECKPOINT_DIRNAME
    resume_checkpoint_dir = output_dir / RESUME_CHECKPOINT_DIRNAME
    tensorboard_dir = output_dir / TENSORBOARD_DIRNAME
    callbacks = [
        _build_loss_checkpoint_callback(
            transformers=transformers,
            output_dir=output_dir,
            best_checkpoint_dir=best_checkpoint_dir,
            resume_checkpoint_dir=resume_checkpoint_dir,
            processor=processor,
            patience=config.early_stop_patience,
            min_delta=config.min_delta,
        )
    ]
    trainer = trl.SFTTrainer(
        model=model,
        args=trl.SFTConfig(
            output_dir=str(output_dir),
            max_steps=config.max_steps,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=config.gradient_checkpointing,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            optim=config.optim,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_ratio=config.warmup_ratio,
            fp16=config.precision == "fp16",
            bf16=config.precision == "bf16",
            logging_steps=1,
            logging_dir=str(tensorboard_dir),
            report_to=["tensorboard"],
            do_eval=eval_dataset is not None,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=config.save_steps if eval_dataset is not None else None,
            save_steps=config.save_steps,
            save_strategy="steps",
            save_total_limit=2,
            max_length=None,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=config.resume_from)
    if not best_checkpoint_dir.exists():
        trainer.save_model(str(best_checkpoint_dir))
    _sync_latest_trainer_checkpoint(output_dir, resume_checkpoint_dir)
    processor.save_pretrained(processor_dir)
    _write_preprocessor(preprocessor_path, config)
    manifest_path = _write_service_manifest(
        output_dir,
        config,
        ready=True,
        checkpoint=best_checkpoint_dir,
        preprocessor=processor_dir,
    )
    latest_manifest_path = manifest_path
    if config.publish_dir is not None:
        latest_manifest_path = _write_service_manifest(
            Path(config.publish_dir),
            config,
            ready=True,
            checkpoint=best_checkpoint_dir,
            preprocessor=processor_dir,
        )
    summary_path = output_dir / SUMMARY_FILENAME
    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = round(float(torch.cuda.max_memory_allocated()) / 1024.0 / 1024.0, 2)
    summary_path.write_text(
        json.dumps(
            {
                "status": "trained",
                "config": asdict(config),
                "train_rows": _count_jsonl_rows(Path(config.dataset)),
                "eval_ratio": config.eval_ratio,
                "adapter_saved": True,
                "peak_vram_mb": peak_vram_mb,
                "checkpoint_path": str(best_checkpoint_dir),
                "resume_checkpoint_path": str(resume_checkpoint_dir),
                "tensorboard_dir": str(tensorboard_dir),
                "preprocessor_path": str(preprocessor_path),
                "manifest_path": str(manifest_path),
                "latest_manifest_path": str(latest_manifest_path),
                "sdpa_backend_report": sdpa_report,
                "torch_compile_report": torch_compile_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


def _build_loss_checkpoint_callback(
    *,
    transformers: Any,
    output_dir: Path,
    best_checkpoint_dir: Path,
    resume_checkpoint_dir: Path,
    processor: Any,
    patience: int,
    min_delta: float,
) -> Any:
    class LossCheckpointCallback(transformers.TrainerCallback):  # type: ignore[misc]
        def __init__(self) -> None:
            self.best_loss = float("inf")
            self.best_step = 0
            self.stale_logs = 0

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
            loss = _logged_loss(logs)
            if loss is None:
                return control
            if loss < self.best_loss - min_delta:
                self.best_loss = loss
                self.best_step = int(getattr(state, "global_step", 0))
                self.stale_logs = 0
                model = kwargs.get("model")
                if model is not None:
                    model.save_pretrained(best_checkpoint_dir)
                    processor.save_pretrained(output_dir / PROCESSOR_DIRNAME)
                    _write_json(
                        output_dir / "best_training_state.json",
                        {"best_loss": self.best_loss, "best_step": self.best_step},
                    )
            else:
                self.stale_logs += 1
                if patience > 0 and self.stale_logs >= patience:
                    control.should_training_stop = True
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            _sync_latest_trainer_checkpoint(output_dir, resume_checkpoint_dir)
            return control

    return LossCheckpointCallback()


def _logged_loss(logs: dict[str, Any] | None) -> float | None:
    if not logs:
        return None
    value = logs.get("eval_loss", logs.get("loss"))
    return float(value) if isinstance(value, (int, float)) else None


def _split_train_eval_dataset(dataset: Any, eval_ratio: float) -> tuple[Any, Any | None]:
    if eval_ratio <= 0 or len(dataset) < 2:
        return dataset, None
    split = dataset.train_test_split(test_size=eval_ratio, seed=42, shuffle=True)
    return split["train"], split["test"]


def _sync_latest_trainer_checkpoint(output_dir: Path, resume_checkpoint_dir: Path) -> None:
    latest = _latest_trainer_checkpoint(output_dir)
    if latest is None:
        return
    if resume_checkpoint_dir.exists():
        shutil.rmtree(resume_checkpoint_dir)
    shutil.copytree(latest, resume_checkpoint_dir)


def _latest_trainer_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = [
        path
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.name.removeprefix("checkpoint-")))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_preprocessor(path: Path, config: TrainingConfig) -> None:
    payload = {
        "model_id": config.model_id,
        "publish_dir": config.publish_dir,
        "training_profile": config.training_profile,
        "training_strategy": config.training_strategy,
        "load_in_4bit": config.load_in_4bit,
        "full_finetune": config.full_finetune,
        "max_steps": config.max_steps,
        "save_steps": config.save_steps,
        "early_stop_patience": config.early_stop_patience,
        "min_delta": config.min_delta,
        "resume_from": config.resume_from,
        "gpu_memory_fraction": config.gpu_memory_fraction,
        "eval_ratio": config.eval_ratio,
        "weight_decay": config.weight_decay,
        "optim": config.optim,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "train_vision_tower": config.train_vision_tower,
        "train_projector": config.train_projector,
        "target_modules": config.target_modules,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "learning_rate": config.learning_rate,
        "precision": config.precision,
        "attn_implementation": config.attn_implementation,
        "image_max_pixels": config.image_max_pixels,
        "flash_attention": config.flash_attention,
        "torch_compile": config.torch_compile,
        "torch_compile_mode": config.torch_compile_mode,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_service_manifest(
    output_dir: Path,
    config: TrainingConfig,
    ready: bool,
    checkpoint: Path | None = None,
    preprocessor: Path | None = None,
) -> Path:
    label_map = {"0": "정상", "1": "노이즈", "2": "표면방전", "3": "코로나방전", "4": "보이드방전"}
    manifest = {
        "task": SERVICE_ARTIFACT_TASK,
        "model_name": config.model_id,
        "model_version": f"local-{config.training_profile}",
        "framework": "transformers",
        "entrypoint": "ml.vlm.src.service_adapter:load_adapter",
        "checkpoint_path": (
            relative_artifact_path(output_dir, checkpoint) if checkpoint is not None else CHECKPOINT_DIRNAME
        ),
        "preprocessor_path": (
            relative_artifact_path(output_dir, preprocessor) if preprocessor is not None else PREPROCESSOR_FILENAME
        ),
        "label_map": label_map,
        "input_spec": {
            "modality": "prpd_png_metadata_evidence",
            "schema_version": "1.0",
            "shape": ["image", "metadata", "model_evidence", "rag_evidence"],
            "dtype": "mixed",
            "notes": "VlmToolInput carries image path and evidence bundle.",
            "ready": ready,
        },
        "output_spec": {
            "schema_version": "1.0",
            "required_fields": ["label_id", "diagnosis", "confidence", "reason", "recommended_action"],
            "notes": "VLM report output schema for runtime fusion.",
        },
        "thresholds": {
            "min_confidence": 0.70,
            "review_confidence": 0.55,
        },
        "runtime": {
            "device": "auto",
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "ready": ready,
            "base_model_id": config.model_id,
            "load_in_4bit": config.load_in_4bit,
            "attn_implementation": config.attn_implementation,
        },
    }
    if ready:
        manifest["ready"] = True
    else:
        manifest["ready"] = False
    manifest_path = output_dir / MODEL_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _count_jsonl_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _load_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError("Training dependencies are missing. Install ml/vlm/requirements.txt, then rerun.") from exc


def _real_training_command(config: TrainingConfig) -> str:
    profile_arg = "" if config.training_profile == "manual" else f"--model-profile {config.training_profile} "
    return "".join(
        [
            "python ml/vlm/train.py ",
            profile_arg,
            f"--model-id {config.model_id} ",
            f"--dataset {config.dataset} ",
            f"--output-dir {config.publish_dir or config.output_dir} ",
            "--skip-dataset-build ",
            f"--attn-implementation {config.attn_implementation} ",
            f"--precision {config.precision} ",
            f"--max-steps {config.max_steps} ",
            f"--save-steps {config.save_steps} ",
            f"--gpu-memory-fraction {config.gpu_memory_fraction} ",
            f"--eval-ratio {config.eval_ratio} ",
            f"--optim {config.optim} ",
            f"--lr-scheduler-type {config.lr_scheduler_type} ",
            f"--warmup-ratio {config.warmup_ratio}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--publish-dir", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--i-understand-8gb-risk", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--gpu-memory-fraction",
        "--gpu-memory-fracion",
        dest="gpu_memory_fraction",
        type=float,
        default=0.9,
    )
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--attn-implementation", default=TORCH_SDPA_ATTENTION, choices=SUPPORTED_ATTENTION_IMPLEMENTATIONS)
    parser.add_argument("--precision", default="fp16", choices=("fp16", "bf16", "off"))
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-mode", default="default", choices=("default", "reduce-overhead", "max-autotune"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    config = build_training_config(
        model_id=args.model_id,
        dataset=args.dataset,
        output_dir=args.output_dir,
        publish_dir=args.publish_dir,
        load_in_4bit=args.load_in_4bit,
        full_finetune=args.full_finetune,
        risk_override=args.i_understand_8gb_risk,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        early_stop_patience=args.early_stop_patience,
        min_delta=args.min_delta,
        resume_from=args.resume_from,
        gpu_memory_fraction=args.gpu_memory_fraction,
        eval_ratio=args.eval_ratio,
        weight_decay=args.weight_decay,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        attn_implementation=args.attn_implementation,
        precision=args.precision,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
    )
    summary_path = write_dry_run_artifacts(config) if args.dry_run else run_sft_training(config)
    print(json.dumps({"summary": str(summary_path), "model_id": config.model_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
