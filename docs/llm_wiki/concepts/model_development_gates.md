# Model Development Gates

## Gate 1: Service Contract Ready

Required before model training becomes the main task:

- Backend service tests pass.
- Frontend typecheck passes.
- Offline mock evaluator writes JSONL and summary output.
- Tool contracts exist for time-series and VLM inference.
- Trace endpoint returns event-level workflow summaries.

## Gate 2: Time-Series Adapter Ready

Required before VLM training context is trusted:

- A checkpoint loader returns `TimeSeriesResult`.
- Validation predictions include probabilities for labels `0` through `4`.
- Confidence calibration is measured.
- Summary features come from `ml/src/features/timeseries_summary.py`.
- False high-confidence errors are reviewed.

## Gate 3: VLM Adapter Ready

Required before agent finalization:

- VLM inference returns parseable JSON.
- VLM output validates against required diagnosis fields.
- Prompt excludes forbidden fields.
- JSON parse success rate and label accuracy are reported.
- Disagreement cases are captured for reviewer tuning.

## Gate 4: Composite Evaluation Ready

Required before claiming system success:

- Offline composite evaluation runs on a validation split.
- Metrics include completed accuracy, needs-review rate,
  false-completed error rate, and TS/VLM agreement rate.
- Reviewer thresholds are documented.
- Human-review branch examples are inspected.
