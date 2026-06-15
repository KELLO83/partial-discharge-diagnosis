# PRD: VLM Report Generator Track

## 1. Purpose

This track builds the multimodal report generator for partial-discharge diagnosis.

The VLM consumes:

```text
PRPD image
+ safe equipment/environment metadata
+ time-series model evidence
+ lightweight vision model evidence
-> strict JSON diagnosis report
```

The VLM is not the only diagnostic model. It explains, formats, and checks evidence from `ml/timeseries` and `ml/vision`.

## 2. Input

Image input:

```text
PRPD PNG from data/manifest.csv manifest.image_path
```

Safe metadata:

```text
equipment_name
equipment_rated_voltage
equipment_rated_current
insulator_type
insulator_name
sensor_type
temperature
humidity
clearance_distance
```

Time-series evidence:

```text
ts_model_name
ts_pred_label_id
ts_confidence
ts_prob_0..ts_prob_4
rms
std
abs_p99
pulse_rate
spectral_energy
```

Vision evidence:

```text
vision_model_name
vision_pred_label_id
vision_confidence
vision_prob_0..vision_prob_4
phase_uniformity_score
band_like_noise_score
ood_score
visual_evidence_summary
```

## 3. Forbidden Prompt Fields

Do not include:

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

`label_id` belongs in assistant targets and evaluation records only.

## 4. Output

Strict JSON:

```json
{
  "label_id": 1,
  "diagnosis": "noise",
  "risk_level": "low",
  "confidence": 0.91,
  "needs_review": false,
  "reason": "The time-series and vision evidence are both more consistent with noise than phase-locked partial discharge.",
  "evidence": [
    "ts_pred_label_id=1",
    "vision_pred_label_id=1",
    "band_like_noise_score=0.88"
  ],
  "recommended_action": "Check sensor contact, grounding, and nearby electromagnetic interference."
}
```

Required fields:

```text
label_id
diagnosis
risk_level
confidence
needs_review
reason
evidence
recommended_action
```

Service adapter contract:

```text
ml/vlm/src/service_adapter.py
load_adapter(context) -> backend
backend.generate_report(VlmToolInput) -> dict
```

The returned dict must contain `label_id`, `confidence`, `diagnosis`, `reason`, and `recommended_action`. The service normalizes this into `VlmResult`.

Allowed diagnoses:

```text
normal
noise
surface_discharge
corona_discharge
void_discharge
unknown
```

## 5. Model Strategy

First candidates:

```text
Qwen/Qwen3-VL-2B-Instruct
Qwen/Qwen2.5-VL-3B-Instruct
HuggingFaceTB/SmolVLM2-2.2B-Instruct
```

Training priority:

1. Frozen VLM inference baseline.
2. JSON-output prompt validation.
3. QLoRA/SFT for report format and domain terminology.
4. Projector/adapter experiments only if needed.
5. Vision encoder LoRA only after time-series + lightweight vision + frozen VLM evidence proves insufficient.

## 6. Evaluation

Required metrics:

```text
json_parse_success_rate
schema_validity_rate
label_accuracy
diagnosis_name_match_rate
forbidden_field_hit_count
hallucinated_field_count
evidence_consistency_rate
needs_review_consistency_rate
```

Review routing checks:

- time-series and vision disagree
- VLM disagrees with high-confidence evidence
- low confidence
- high OOD score
- invalid JSON

## 7. Non-Goals

- Do not place raw CSV rows in prompts.
- Do not train the full vision tower first.
- Do not make VLM output override evidence without review.
- Do not treat report quality as classification quality.

## 8. Current CLI

The standard entrypoint builds the instruction JSONL and then runs dry-run or QLoRA/SFT:

```powershell
python ml/vlm/train.py --model-profile smolvlm2_2b_qlora --manifest data/manifest.csv --sample-size 20 --dry-run
```

Actual adapter training:

```powershell
python ml/vlm/train.py --model-profile qwen3_vl_2b_qlora --manifest data/manifest.csv --sample-size 500 --max-steps 100
```

Available model profiles:

```text
qwen3_vl_2b_qlora
  Default Qwen3-VL 2B profile. 4-bit QLoRA, frozen vision tower, frozen projector, 512x512 image budget.

qwen2_5_vl_3b_qlora
  Stable fallback profile when Qwen3-VL support is unstable or quality comparison is needed.

smolvlm2_2b_qlora
  Low-VRAM smoke profile. Use before longer Qwen-family runs.
```

Input construction:

```text
data/manifest.csv
-> resolved PRPD PNG path
-> safe metadata prompt
-> optional time-series context CSV
-> instruction_dataset.jsonl
-> VLM SFT / QLoRA
```

Default outputs:

```text
artifacts/models/vlm/instruction_dataset.jsonl
artifacts/models/vlm/training_config.json
artifacts/models/vlm/dry_run_summary.json
```
