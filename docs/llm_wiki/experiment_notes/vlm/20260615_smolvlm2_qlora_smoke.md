# 2026-06-15 SmolVLM2 QLoRA Smoke Baseline

## Purpose

Validate that a real VLM training loop, checkpoint artifact, and service adapter
path work locally.

## Run

```text
base_model: HuggingFaceTB/SmolVLM2-2.2B-Instruct
model_profile: smolvlm2_2b_qlora
checkpoint: artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/best.pt
manifest: artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/model_manifest.json
attention_backend: sdpa
gpu_memory_fraction: 0.9
```

## Eval Loss

| Step | Eval loss |
|---:|---:|
| 5 | 10.6423 |
| 10 | 7.1014 |
| 15 | 4.8421 |
| 20 | 4.2387 |

## Train Loss

```text
step_1: 13.6764
step_20: 4.2576
```

## Notes

- This run proves the VLM path is no longer mock-only.
- It is a smoke baseline, not a final quality claim.
- Next work is a larger validation run with JSON parse rate, schema validity,
  label accuracy, macro F1, and forbidden-field leakage checks.
