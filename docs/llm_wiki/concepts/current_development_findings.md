# Current Development Findings

## Current Claim

The project direction is clear: build a composite industrial diagnosis system,
not a single image classifier.

The current implementation now has enough service scaffolding that future model
development should focus on replacing adapters rather than changing service
contracts.

## Stable So Far

- The time-series task is 5-class classification from `(20, 7680)` CSV signals.
- VLM input should use PRPD image, safe metadata, time-series prediction, and
  summary features.
- Raw CSV values must not be placed in Agent traces or VLM prompts.
- The service reviewer owns `completed`, `needs_review`, and `rejected` routing.
- Upload artifacts are stored with checksums and passed to tool contracts.
- The frontend can submit diagnosis inputs and display trace events.

## Current Adapter Replacement Points

```text
TimeSeriesInferenceAdapter.run(input) -> TimeSeriesResult
VlmInferenceAdapter.run(input) -> VlmResult
```

The model developer should implement these two adapters first. Optional vision
classification should wait until the time-series + VLM + reviewer loop is
evaluated.

## Active Risks

- Time-series confidence must be calibrated before reviewer thresholds are final.
- VLM JSON quality is not proven until QLoRA smoke runs and parse-rate
  evaluation exist.
- `openai-agents` is declared as a service dependency, but the deterministic
  local runtime remains the default path until SDK orchestration is implemented
  and tested.
- Optional external time-series repositories may require additional packages
  such as `reformer-pytorch` or `tsai`.

## Next Useful Work

1. Implement real time-series checkpoint inference adapter.
2. Export validation-set time-series predictions for VLM context.
3. Run VLM JSON-output smoke training.
4. Run offline composite evaluation with real TS predictions and mock/real VLM.
5. Tune reviewer thresholds based on false-completed error rate.
