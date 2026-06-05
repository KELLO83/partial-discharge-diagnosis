from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class TrainingRiskError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model_id: str
    dataset: str
    output_dir: str
    load_in_4bit: bool
    full_finetune: bool
    max_steps: int
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    train_vision_tower: bool = False
    train_projector: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    precision: str = "fp16"
    image_max_pixels: str = "512x512"
    flash_attention: bool = False


def build_training_config(
    model_id: str,
    dataset: str,
    output_dir: str,
    load_in_4bit: bool,
    full_finetune: bool,
    risk_override: bool,
    max_steps: int,
) -> TrainingConfig:
    if full_finetune and not risk_override:
        raise TrainingRiskError("Full fine-tuning is blocked for RTX 4060 Laptop 8GB without explicit override.")
    if not load_in_4bit and not risk_override:
        raise TrainingRiskError("VLM SFT on 8GB must use --load-in-4bit unless --i-understand-8gb-risk is set.")
    if max_steps < 1:
        raise ValueError("--max-steps must be at least 1.")
    return TrainingConfig(
        model_id=model_id,
        dataset=dataset,
        output_dir=output_dir,
        load_in_4bit=load_in_4bit,
        full_finetune=full_finetune,
        max_steps=max_steps,
    )


def write_dry_run_artifacts(config: TrainingConfig) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "training_config.json"
    summary_path = output_dir / "dry_run_summary.json"
    config_payload = asdict(config)
    train_rows = _count_jsonl_rows(Path(config.dataset))
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "status": "dry_run_ready",
                "config": config_payload,
                "train_rows": train_rows,
                "valid_rows": train_rows,
                "adapter_saved": False,
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
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    processor = transformers.AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    train_dataset = datasets.load_dataset("json", data_files=config.dataset, split="train")
    peft_config = None
    if not config.full_finetune:
        peft_config = peft.LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
    trainer = trl.SFTTrainer(
        model=model,
        args=trl.SFTConfig(
            output_dir=str(output_dir),
            max_steps=config.max_steps,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=config.gradient_checkpointing,
            learning_rate=config.learning_rate,
            fp16=config.precision == "fp16",
            bf16=config.precision == "bf16",
            logging_steps=1,
            save_steps=config.max_steps,
        ),
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    summary_path = output_dir / "train_summary.json"
    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = round(float(torch.cuda.max_memory_allocated()) / 1024.0 / 1024.0, 2)
    summary_path.write_text(
        json.dumps(
            {
                "status": "trained",
                "config": asdict(config),
                "train_rows": _count_jsonl_rows(Path(config.dataset)),
                "adapter_saved": True,
                "peak_vram_mb": peak_vram_mb,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


def _count_jsonl_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _load_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError("Training dependencies are missing. Install vlm/requirements.txt, then rerun.") from exc


def _real_training_command(config: TrainingConfig) -> str:
    return (
        "python vlm/scripts/train_sft.py "
        f"--model-id {config.model_id} "
        f"--dataset {config.dataset} "
        f"--output-dir {config.output_dir} "
        "--load-in-4bit "
        f"--max-steps {config.max_steps}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--i-understand-8gb-risk", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_training_config(
        model_id=args.model_id,
        dataset=args.dataset,
        output_dir=args.output_dir,
        load_in_4bit=args.load_in_4bit,
        full_finetune=args.full_finetune,
        risk_override=args.i_understand_8gb_risk,
        max_steps=args.max_steps,
    )
    summary_path = write_dry_run_artifacts(config) if args.dry_run else run_sft_training(config)
    print(json.dumps({"summary": str(summary_path), "model_id": config.model_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
