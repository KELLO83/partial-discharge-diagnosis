# 2026-06-15 EfficientNet-B0 Vision Baseline

## Purpose

Create a lightweight PRPD image classifier baseline that can provide compact
vision context to the composite diagnosis workflow and VLM prompts.

## Run

```text
model: efficientnet_b0
checkpoint: artifacts/models/vision/efficientnet_b0/20260615_201805/best.pt
manifest: artifacts/models/vision/model_manifest.json
```

## Metrics

```text
best_valid_accuracy: 1.0000
best_valid_loss: 0.000044218
final_valid_loss: 0.0000938665
```

## Notes

- The metric is very high on the current local split. Treat it as a baseline
  result until split leakage, class balance, and calibration are reviewed.
- The vision model is not the VLM. It is a separate PRPD image classifier whose
  prediction can be provided to the VLM as text context.
