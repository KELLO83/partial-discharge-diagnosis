from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.vlm.src.model_profiles import profile_keys, resolve_training_profile
from ml.vlm.scripts.build_instruction_dataset import build_instruction_dataset
from ml.vlm.scripts.train_sft import build_training_config, run_sft_training, write_dry_run_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PRPD instruction data and train the VLM SFT adapter.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--dataset-output", type=Path, default=Path("artifacts/models/vlm/instruction_dataset.jsonl"))
    parser.add_argument("--ts-context", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--skip-dataset-build", action="store_true")
    parser.add_argument("--model-profile", default="qwen2_5_vl_3b_qlora", choices=profile_keys())
    parser.add_argument("--model-id", default=None, help="Override the model id from --model-profile.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models/vlm"))
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--i-understand-8gb-risk", action="store_true")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=None)
    parser.add_argument("--attn-implementation", default=None, choices=("sdpa", "eager", "flash_attention_2"))
    parser.add_argument("--precision", default=None, choices=("fp16", "bf16", "off"))
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-mode", default="default", choices=("default", "reduce-overhead", "max-autotune"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    profile = resolve_training_profile(args.model_profile)
    dataset_path = _dataset_path(args)
    dataset_rows = _prepare_dataset(args, dataset_path)
    training_config = build_training_config(
        model_id=args.model_id or profile.model_id,
        dataset=str(dataset_path),
        output_dir=str(args.output_dir),
        load_in_4bit=profile.load_in_4bit if args.load_in_4bit is None else args.load_in_4bit,
        full_finetune=args.full_finetune,
        risk_override=args.i_understand_8gb_risk,
        max_steps=args.max_steps,
        attn_implementation=args.attn_implementation or profile.attn_implementation,
        precision=args.precision or profile.precision,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
    )
    training_config = replace(
        training_config,
        training_profile=profile.key,
        training_strategy=profile.training_strategy,
        batch_size=profile.batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        gradient_checkpointing=profile.gradient_checkpointing,
        train_vision_tower=profile.train_vision_tower,
        train_projector=profile.train_projector,
        lora_r=profile.lora_r,
        lora_alpha=profile.lora_alpha,
        lora_dropout=profile.lora_dropout,
        learning_rate=profile.learning_rate,
        target_modules=profile.target_modules,
        image_max_pixels=profile.image_max_pixels,
    )
    summary_path = write_dry_run_artifacts(training_config) if args.dry_run else run_sft_training(training_config)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "dataset": str(dataset_path),
                "dataset_rows_written": dataset_rows,
                "model_id": training_config.model_id,
            },
            ensure_ascii=False,
        )
    )


def _dataset_path(args: argparse.Namespace) -> Path:
    return args.dataset if args.dataset is not None else args.dataset_output


def _prepare_dataset(args: argparse.Namespace, dataset_path: Path) -> int | None:
    if args.skip_dataset_build:
        if not dataset_path.exists():
            raise FileNotFoundError(f"VLM dataset does not exist: {dataset_path}")
        return None
    summary = build_instruction_dataset(
        manifest_path=args.manifest,
        output_path=dataset_path,
        sample_size=args.sample_size,
        ts_context_path=args.ts_context,
    )
    return summary.rows_written


if __name__ == "__main__":
    main()
