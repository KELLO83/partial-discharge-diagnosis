# VLM Development Runbook

This runbook is the short operational version of `docs/VLM_TRAINING_GUIDE.md`.
Use the guide for detailed explanations; use this file when re-running or
checking the current VLM baseline.

## Current Baseline

| Item | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |
| Training method | 4-bit QLoRA SFT |
| Attention backend | PyTorch SDPA |
| Default GPU memory target | `--gpu-memory-fraction 0.9` |
| Output directory | `artifacts/models/vlm/smolvlm2_2b_qlora/<run_id>/` |
| Active checkpoint | `artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/best.pt` |

Qwen-VL remains a future comparison candidate. The first working local baseline
is SmolVLM2 because it completed on the available RTX 4060 Laptop workflow with
the existing training stack.

## Input Contract

The VLM receives one PRPD image plus text context. It is not trained from raw
CSV arrays and it does not require the time-series or vision classifiers to be
trained first.

```text
PRPD image
+ safe equipment/environment metadata
+ time-series context CSV
+ optional vision context CSV
-> SmolVLM2 QLoRA adapter
-> structured diagnosis JSON/report context
```

Forbidden prompt fields remain forbidden even when they exist in the manifest:

```text
label_id
label_name
PD_type
sample_id
image_path string
timeseries_path string
json_path string
file name
defect_details
defect_nums
max_discharge_value
full raw CSV values
```

## Build Context CSVs

```powershell
python ml/vlm/scripts/export_ts_context.py `
  --manifest data/manifest.csv `
  --output artifacts/models/vlm/context/ts_context.csv

python ml/vlm/scripts/export_vision_context.py `
  --manifest data/manifest.csv `
  --output artifacts/models/vlm/context/vision_context.csv
```

The context CSVs contain compact model predictions and summary features that are
safe to place in text prompts. They are not user-facing reports.

## Train

Smoke run:

```powershell
python ml/vlm/train.py `
  --model-profile smolvlm2_2b_qlora `
  --manifest data/manifest.csv `
  --sample-size 20 `
  --ts-context artifacts/models/vlm/context/ts_context.csv `
  --vision-context artifacts/models/vlm/context/vision_context.csv `
  --epochs 1 `
  --max-steps 20 `
  --eval-steps 5 `
  --gpu-memory-fraction 0.9 `
  --output-dir artifacts/models/vlm/smolvlm2_2b_qlora
```

Longer local run:

```powershell
python ml/vlm/train.py `
  --model-profile smolvlm2_2b_qlora `
  --manifest data/manifest.csv `
  --sample-size 2000 `
  --ts-context artifacts/models/vlm/context/ts_context.csv `
  --vision-context artifacts/models/vlm/context/vision_context.csv `
  --epochs 2 `
  --eval-steps 100 `
  --gpu-memory-fraction 0.9 `
  --output-dir artifacts/models/vlm/smolvlm2_2b_qlora
```

## Check Results

Each run directory should contain:

```text
best.pt
model_manifest.json
summary.json
events.out.tfevents.*
```

Open TensorBoard from the run parent:

```powershell
tensorboard --logdir artifacts/models/vlm/smolvlm2_2b_qlora
```

Current smoke metrics from `20260615_202950`:

| Step | Eval loss |
|---:|---:|
| 5 | 10.6423 |
| 10 | 7.1014 |
| 15 | 4.8421 |
| 20 | 4.2387 |

The run proves the training loop and adapter path are usable. It is not yet a
final production-quality VLM.

## Activate In Service

Set root `.env`:

```dotenv
MODEL_ADAPTER_MODE=checkpoint
MODEL_VLM_MANIFEST=artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/model_manifest.json
MODEL_VLM_CHECKPOINT=artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/best.pt
```

The frontend reads this only through backend APIs. It does not load the VLM
checkpoint directly.
