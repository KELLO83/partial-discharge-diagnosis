# PRD: Pre-Model Composite Diagnosis Infrastructure

## 1. Purpose

Before training the next time-series or VLM model, the project needs stable
contracts and deterministic scaffolding for the final composite diagnosis
system.

The target system is:

```text
CSV time-series inference
+ PRPD image
+ safe equipment/environment metadata
+ VLM JSON diagnosis
+ Agent reviewer and guardrails
-> final diagnosis or human-review decision
```

This stage does not optimize model accuracy. It prepares the code that lets
future models plug into the same service and evaluation workflow.

## 2. Required Contracts

### Time-Series Tool Output

```json
{
  "model_name": "mock_patchtst",
  "model_version": "pre_model_mock",
  "label_id": 3,
  "label_name": "�ڷγ� ����",
  "confidence": 0.87,
  "probabilities": {"0": 0.02, "1": 0.04, "2": 0.06, "3": 0.87, "4": 0.01},
  "features": {
    "rms": 30.37,
    "std": 4.96,
    "abs_p99": 39.0,
    "pulse_rate": 0.0069,
    "spectral_energy": 13982100.0
  }
}
```

### VLM Tool Output

```json
{
  "model_name": "mock_qwen3_vl_2b",
  "model_version": "pre_model_mock",
  "label_id": 3,
  "diagnosis": "�ڷγ� ����",
  "risk_level": "����",
  "confidence": 0.89,
  "reason": "...",
  "recommended_action": "..."
}
```

### Reviewer Output

The reviewer must return one of:

```text
completed
needs_review
rejected
```

The reviewer must not invent a new label. It may only choose from validated
tool outputs.

## 3. Guardrail Rules

- Reject invalid input before any model/tool call.
- Mark `needs_review` if time-series confidence is below `0.60`.
- Mark `needs_review` if VLM confidence is below `0.60`.
- Mark `needs_review` if time-series and VLM labels disagree.
- Mark `needs_review` if probabilities are missing, invalid, or do not sum
  close to `1.0`.
- Keep raw CSV values, image bytes, sample IDs, file paths, labels, and defect
  details out of VLM prompts and traces.

## 4. Required Pre-Model Code

1. Shared partial-discharge label and action policy.
2. Shared time-series summary feature extractor.
3. Deterministic mock time-series and VLM tools that satisfy the final schema.
4. Reviewer guardrail code independent from the model implementation.
5. Offline diagnosis evaluator that can run the composite workflow without
   training real models.
6. Tests for low-confidence, label-disagreement, invalid-probability, and
   offline-evaluation paths.
7. Upload artifact storage with checksums and safe per-diagnosis paths.
8. Tool input contracts for time-series, VLM, and optional vision inference.
9. Agent runtime adapter that supports deterministic local execution now and
   can be replaced with OpenAI Agents SDK orchestration later.
10. Trace events that record validation, tool calls, reviewer decisions, and
    final report decisions without storing raw CSV rows or image bytes.

## 5. Model Development Gate

Start model development only after:

- `pytest service/backend/tests ml/vlm/tests/test_export_ts_context.py` passes.
- The offline evaluator writes a summary JSON.
- Time-series context export uses the same summary feature code as service
  tools.
- The service returns deterministic `completed`, `needs_review`, and
  `rejected` branches in tests.
- `/diagnose` stores upload artifacts and passes stable file paths into tool
  contracts.
- `/diagnose/{id}/trace` returns trace events with validation/tool/reviewer
  summaries.

## 5.1 Adapter Replacement Points

After this stage, model development should only replace these adapters:

```text
TimeSeriesInferenceAdapter.run(input) -> TimeSeriesResult
VlmInferenceAdapter.run(input) -> VlmResult
OptionalVisionInferenceAdapter.run(input) -> VisionResult
```

Do not change the FastAPI request shape, reviewer rules, trace schema, or VLM
prompt safety rules when swapping model implementations.

## 6. Follow-Up After Models Exist

After the first production-like time-series checkpoint exists:

1. Replace the mock time-series tool with checkpoint inference.
2. Add calibration metadata and ECE/reliability reporting.
3. Export validation-set time-series predictions into VLM training context.
4. Run offline composite evaluation against validation labels.
5. Tune reviewer thresholds before connecting the React UI to real inference.

## 7. Optional PRPD Vision Model Track

The final product direction remains VLM-based composite diagnosis. A standalone
PRPD vision classifier is not required for the first complete system because
the VLM already consumes the PRPD image and generates structured diagnosis.

However, a lightweight vision model may be useful as an auxiliary evidence
source.

### When It Is Worth Building

- The VLM image understanding is unstable on PRPD plots.
- The reviewer needs an independent image-only sanity check.
- The service needs a cheaper fallback when VLM inference is unavailable.
- Offline evaluation shows repeated time-series/VLM disagreement.
- The project needs explainable evidence such as image-only confidence,
  confusion matrix, or saliency/attention inspection.

### When To Defer It

- Time-series inference is not yet stable.
- VLM instruction dataset quality is not validated.
- The composite offline evaluator is not running.
- The only goal is to generate a field-engineer diagnosis report.

### Scope

The vision model is an auxiliary classifier, not the main product.

Input:

```text
PRPD PNG image
```

Output:

```json
{
  "model_name": "prpd_vision_classifier",
  "model_version": "v0",
  "label_id": 3,
  "label_name": "�ڷγ� ����",
  "confidence": 0.81,
  "probabilities": {"0": 0.03, "1": 0.05, "2": 0.08, "3": 0.81, "4": 0.03}
}
```

Recommended candidates:

```text
EfficientNetV2-S
ConvNeXt-Tiny
Swin-Tiny
ViT-Small
```

Start with ImageNet-pretrained `timm` models and a 5-class classification head.
Do not train from scratch unless pretrained models fail.

### Integration Into Composite Diagnosis

If implemented, the reviewer should treat the vision classifier as supporting
evidence:

```text
time-series result
+ VLM result
+ optional image-only classifier result
-> reviewer decision
```

Branching examples:

- If VLM and vision agree but time-series disagrees, return `needs_review`.
- If time-series and VLM agree but vision disagrees with low confidence, still
  allow `completed`.
- If all three disagree, return `needs_review`.
- If VLM is unavailable, use time-series + vision as a reduced route.

### Development Order

1. Finish time-series inference contract.
2. Finish VLM instruction dataset and JSON evaluation.
3. Run composite offline evaluator.
4. Add a simple image-only vision baseline only if it answers a concrete
   failure mode.

### Required Metrics

- image-only accuracy
- macro F1
- per-class recall
- calibration/ECE
- disagreement rate with time-series model
- disagreement rate with VLM

This track should not delay the main composite diagnosis workflow.
