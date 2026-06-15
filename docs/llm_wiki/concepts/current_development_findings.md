# Current Development Findings

## Current Claim

The project direction is clear: build a composite industrial diagnosis system,
not a single image classifier.

The implementation has moved past pure mock adapters. Time-series, vision, and
VLM baseline checkpoints can now be selected through the backend artifact
registry and root `.env` overrides.

## Stable So Far

- The time-series task is 5-class classification from `(20, 7680)` CSV signals.
- The active time-series baseline is `inception_time_small`.
- The active PRPD image baseline is `efficientnet_b0`.
- The active VLM baseline is `smolvlm2_2b_qlora`.
- VLM input should use PRPD image, safe metadata, compact time-series context,
  and optional compact vision context.
- Raw CSV values must not be placed in Agent traces or VLM prompts.
- The service reviewer owns `completed`, `needs_review`, and `rejected` routing.
- Upload artifacts are stored with checksums and passed to tool contracts.
- The frontend can submit diagnosis inputs and display trace events, reports,
  similar cases, and model status.

## Current Adapter Replacement Points

```text
TimeSeriesInferenceAdapter.run(input) -> TimeSeriesResult
VisionInferenceAdapter.run(input) -> VisionResult
VlmInferenceAdapter.run(input) -> VlmResult
```

The current development path is no longer "implement adapters first." The next
work is to evaluate and harden the checkpoint-backed adapters that now exist.

## Active Risks

- Time-series and vision confidence must be calibrated before reviewer
  thresholds are final.
- The VLM smoke run proves the training and serving path, but a larger
  validation run is still needed before quality claims.
- Composite offline evaluation must verify completed accuracy, needs-review
  rate, false-completed error rate, and TS/VLM agreement rate.
- OpenAI Agents SDK is not required for the current diagnosis path. The local
  deterministic runtime remains the default until SDK orchestration is designed
  and tested.

## Next Useful Work

1. Run larger validation for `inception_time_small` and `efficientnet_b0`.
2. Run a larger SmolVLM2 QLoRA experiment with held-out JSON evaluation.
3. Export stable TS and vision context CSVs for VLM training.
4. Run offline composite evaluation with checkpoint-backed models.
5. Tune reviewer thresholds based on false-completed error rate.
