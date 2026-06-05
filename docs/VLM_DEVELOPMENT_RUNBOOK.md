# VLM Development Runbook

## Goal

VLM experiments are a separate track from time-series classification experiments. The input is a PRPD image, leakage-safe equipment/environment metadata, and time-series summary values. The output is structured diagnosis JSON.

## Model Order

1. `Qwen/Qwen3-VL-2B-Instruct`: first QLoRA target for RTX 4060 Laptop 8GB.
2. `Qwen/Qwen2.5-VL-3B-Instruct`: stable fallback if local Qwen3-VL support is unstable.
3. `Qwen/Qwen3-VL-4B-Instruct`: try only after 2B/3B smoke tests succeed and VRAM usage is measured.

## Build Data

```bash
python vlm/scripts/export_ts_context.py --manifest Train/manifest.csv --sample-size 20 --output .omo/evidence/vlm-task-5-ts-context.csv
python vlm/scripts/build_instruction_dataset.py --manifest Train/manifest.csv --sample-size 20 --ts-context .omo/evidence/vlm-task-5-ts-context.csv --output .omo/evidence/vlm-smoke.jsonl
python vlm/scripts/validate_instruction_dataset.py --input .omo/evidence/vlm-smoke.jsonl --output .omo/evidence/vlm-task-7-validate.json
```

## Dry-Run Pipeline

```bash
python vlm/scripts/run_inference.py --dataset .omo/evidence/vlm-smoke.jsonl --model-id Qwen/Qwen3-VL-2B-Instruct --load-in-4bit --limit 20 --dry-run --output .omo/evidence/vlm-task-8-predictions.jsonl
python vlm/scripts/evaluate_outputs.py --predictions .omo/evidence/vlm-task-8-predictions.jsonl --output .omo/evidence/vlm-task-7-evaluation.json
python vlm/scripts/train_sft.py --dataset .omo/evidence/vlm-smoke.jsonl --model-id Qwen/Qwen3-VL-2B-Instruct --output-dir .omo/evidence/vlm-task-9-adapter --load-in-4bit --max-steps 10 --dry-run
```

## Real Training

Install VLM dependencies first:

```bash
pip install -r vlm/requirements.txt
```

Then run:

```bash
python vlm/scripts/train_sft.py --dataset .omo/evidence/vlm-smoke.jsonl --model-id Qwen/Qwen3-VL-2B-Instruct --output-dir results/vlm/qwen3_vl_2b_lora --load-in-4bit --max-steps 10
```

Keep these first-run constraints:

- batch size 1
- 4-bit NF4
- LoRA rank 8
- gradient checkpointing enabled
- vision tower frozen
- flash attention disabled by default on Windows

## Interpretation

Dry-run inference echoes assistant targets and is only for validating JSON parsing, leakage checks, evaluation code, and file paths. It is not model performance. Real performance is measured only after running without `--dry-run`.

## Prompt Shape

```text
You are an assistant model for partial-discharge diagnosis in industrial power equipment.
Use only the provided PRPD image and text information.
Do not guess. Output JSON only.

[Equipment and environment information]
equipment_name: ACSR-OC
equipment_rated_voltage: 22900V
equipment_rated_current: 268A
sensor_type: HFCT
temperature: 19
humidity: 66

[Time-series model analysis]
ts_model_name: feature_summary_untrained
ts_pred_class: unavailable
ts_confidence: unavailable
rms: 30.3781
std: 4.96528
abs_p99: 39
pulse_rate: 0.00693359
spectral_energy: 1.39821e+07
```
