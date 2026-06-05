# VLM Development Runbook

## Goal

VLM 실험은 시계열 분류 실험과 별도 트랙이다. 입력은 PRPD 이미지, 누수 없는 설비/환경 메타데이터, 시계열 요약값이고 출력은 구조화된 진단 JSON이다.

## Model Order

1. `Qwen/Qwen3-VL-2B-Instruct`: 8GB RTX 4060 Laptop의 첫 QLoRA 대상.
2. `Qwen/Qwen2.5-VL-3B-Instruct`: Qwen3-VL 로컬 지원이 불안정하면 stable fallback.
3. `Qwen/Qwen3-VL-4B-Instruct`: 2B/3B 스모크 성공 후 VRAM 측정이 있을 때만 시도.

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

Install the VLM dependencies first:

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
당신은 산업 전력설비 부분방전 진단 보조 모델입니다.
제공된 PRPD 이미지와 텍스트 정보만 사용하세요.
추측하지 말고 반드시 JSON만 출력하세요.

[설비 및 환경 정보]
equipment_name: ACSR-OC
equipment_rated_voltage: 22900V
equipment_rated_current: 268A
sensor_type: HFCT
temperature: 19
humidity: 66

[시계열 모델 분석]
ts_model_name: feature_summary_untrained
ts_pred_class: unavailable
ts_confidence: unavailable
rms: 30.3781
std: 4.96528
abs_p99: 39
pulse_rate: 0.00693359
spectral_energy: 1.39821e+07
```
