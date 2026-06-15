# Model Development Gates

## Gate 1: Service Contract Ready

Status: complete for the local diagnosis workflow.

Required before model training becomes the main task:

- Backend service tests pass.
- Frontend typecheck passes.
- Offline mock evaluator writes JSONL and summary output.
- Tool contracts exist for time-series and VLM inference.
- Trace endpoint returns event-level workflow summaries.

## Gate 2: Time-Series Adapter Ready

Status: baseline complete, calibration still required.

Required before VLM training context is trusted:

- A checkpoint loader returns `TimeSeriesResult`.
- Validation predictions include probabilities for labels `0` through `4`.
- Confidence calibration is measured.
- Summary features come from `ml/timeseries/src/features/timeseries_summary.py`.
- False high-confidence errors are reviewed.

Current baseline:

```text
model: inception_time_small
checkpoint: artifacts/models/time_series/inception_time_small/20260615_194838/best.pt
```

## Gate 3: VLM Adapter Ready

Status: smoke baseline complete, larger evaluation still required.

Required before agent finalization:

- VLM inference returns parseable JSON.
- VLM output validates against required diagnosis fields.
- Prompt excludes forbidden fields.
- JSON parse success rate and label accuracy are reported.
- Disagreement cases are captured for reviewer tuning.

Current baseline:

```text
model: smolvlm2_2b_qlora
checkpoint: artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/best.pt
```

## Gate 3b: Vision Adapter Ready

Status: baseline complete, calibration still required.

Required before relying on PRPD image classifier context:

- A checkpoint loader returns a vision result.
- Validation predictions include probabilities for labels `0` through `4`.
- Confidence calibration is measured.
- False high-confidence errors are reviewed.

Current baseline:

```text
model: efficientnet_b0
checkpoint: artifacts/models/vision/efficientnet_b0/20260615_201805/best.pt
```

## Gate 4: Composite Evaluation Ready

Status: next required milestone.

Required before claiming system success:

- Offline composite evaluation runs on a validation split.
- Metrics include completed accuracy, needs-review rate,
  false-completed error rate, and TS/VLM agreement rate.
- Reviewer thresholds are documented.
- Human-review branch examples are inspected.
