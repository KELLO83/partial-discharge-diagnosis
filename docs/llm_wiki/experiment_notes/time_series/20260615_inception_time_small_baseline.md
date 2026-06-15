# 2026-06-15 InceptionTime-small Baseline

## Purpose

Create a lightweight checkpoint-backed time-series classifier baseline for the
service runtime.

## Run

```text
model: inception_time_small
checkpoint: artifacts/models/time_series/inception_time_small/20260615_194838/best.pt
manifest: artifacts/models/time_series/inception_time_small/20260615_194838/model_manifest.json
```

## Metrics

```text
best_epoch: 17
best_valid_accuracy: 0.9900
best_macro_f1: 0.989959
best_train_loss: 0.03295
final_train_loss: 0.03146
```

## Notes

- This is a strong local baseline, but confidence calibration still needs a
  dedicated review before reviewer thresholds are treated as final.
- Frontend model status reads this through backend APIs, not by opening the
  checkpoint file directly.
