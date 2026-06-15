# VLM Implementation Plan

## Goal

The VLM stage is not PRPD image-only classification. The goal is to build a model that receives the following inputs and generates structured diagnosis JSON:

```text
PRPD image
+ safe equipment/environment metadata
+ optional vision model context
+ time-series model prediction
+ time-series summary features
-> lightweight pretrained VLM
-> diagnosis JSON
```

The target GPU is an RTX 4060 Laptop with 8GB VRAM. Full fine-tuning of large VLMs is excluded. Fine-tune a lightweight pretrained VLM with LoRA/QLoRA.

## Model Selection

### Current Implemented Baseline: HuggingFaceTB/SmolVLM2-2.2B-Instruct

The current local VLM baseline is `smolvlm2_2b_qlora`.

Reasons:

- It is small enough for the RTX 4060 Laptop 8GB workflow with 4-bit QLoRA.
- It supports the project input shape: PRPD image plus text context.
- It completed a local smoke run and produced a checkpoint-backed service artifact.
- It can use PyTorch SDPA without requiring FlashAttention.

Current artifact:

```text
artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/best.pt
```

### Future Comparison Candidates

- `Qwen/Qwen2.5-VL-3B-Instruct`: strong comparison candidate after the current
  SmolVLM2 baseline is evaluated on a larger validation set.
- `Qwen/Qwen3-VL-2B-Instruct`: keep as a target if the local Transformers stack
  and Windows dependency path are stable.
- `google/paligemma2-3b-mix-224`: possible image-text transfer candidate, but
  less aligned with strict instruction-following JSON reports.

### Not Current Targets

- `Qwen/Qwen3-VL-4B-Instruct`: high OOM risk on 8GB VRAM.
- `LGAI-EXAONE/EXAONE-4.0-1.2B-AWQ`: text-only LLM, not a VLM training target.
- Large EXAONE VLM variants: too heavy for the current 8GB local training target.

## Input Data Design

Do not feed raw CSV directly into the VLM. Use one image and one text context.

### Strategy A: Default Input

```text
image:
  PRPD PNG from manifest.image_path

text:
  equipment information
  environment information
  optional vision model analysis
  time-series model analysis
  JSON-output instruction
```

### Image Input

```text
image_path from data/manifest.csv
-> PRPD PNG
-> VLM processor image input
```

Do not include the `image_path` string itself in prompt text because file paths or file names may contain labels.

### Safe Metadata for Text Input

```text
equipment_name
equipment_rated_voltage
equipment_rated_current
insulator_type / insulator_name
sensor_type
temperature
humidity
clearance_distance
```

### Vision and Time-Series Model Information for Text Input

The VLM can be trained without pre-training the vision and time-series
classifiers, but the preferred project flow is to include compact model context
when it exists. Export the following information into separate CSVs and join it
in the VLM dataset builder:

Vision context:

```text
sample_id
vision_model_name
vision_pred_label_id
vision_confidence
vision_prob_0
vision_prob_1
vision_prob_2
vision_prob_3
vision_prob_4
```

Time-series context:

```text
sample_id
ts_model_name
ts_pred_label_id
ts_confidence
ts_prob_0
ts_prob_1
ts_prob_2
ts_prob_3
ts_prob_4
rms
std
abs_p99
pulse_rate
spectral_energy
```

If model results are unavailable, the smoke dataset should explicitly state that
the model result is unavailable and use feature-only context.

### Fields Forbidden in Prompts

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

`label_id` is used for targets and evaluation, but never in user prompt text.

## Output Format

Initial VLM output should be strict JSON rather than free-form natural language.

```json
{
  "label_id": 3,
  "diagnosis": "corona_discharge",
  "risk_level": "caution",
  "reason": "The PRPD pattern and time-series summary features are consistent with corona discharge.",
  "recommended_action": "Inspect high-voltage insulation areas and monitor whether discharge signals increase."
}
```

Evaluation items:

- JSON parse success rate
- schema validity
- label accuracy
- macro F1
- confusion matrix
- hallucinated field count
- forbidden prompt field leakage count

## Training Method

### Default Method

```text
pretrained VLM
-> 4-bit QLoRA
-> SFT
-> strict JSON diagnosis generation
```

### Default 8GB Settings

```yaml
model_id: HuggingFaceTB/SmolVLM2-2.2B-Instruct
quantization: 4bit_nf4
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: all-linear
train_vision_tower: false
train_projector: false
batch_size: 1
gradient_accumulation_steps: 8
gradient_checkpointing: true
max_length: null
image_max_pixels: 512x512
attention_backend: sdpa
gpu_memory_fraction: 0.9
```

Do not train the vision tower at first. Let the pretrained vision encoder extract generic points, lines, density, and distribution patterns from PRPD images, and apply LoRA to language layers to learn the JSON diagnosis format.

## Implementation Order

### 1. Define the VLM Input Contract

Files:

```text
ml/vlm/src/schema.py
ml/vlm/src/prompts.py
ml/vlm/tests/test_prompts.py
```

Tasks:

- Define the safe metadata whitelist.
- Define forbidden prompt fields.
- Define the target JSON schema.
- Define the image + text message format.
- Add prompt leakage tests.

### 2. Export Time-Series Context

File:

```text
ml/vlm/scripts/export_ts_context.py
```

Tasks:

- Generate per-sample feature/context from `data/manifest.csv`.
- Join prediction/probability columns if time-series model results exist.
- Do not export raw CSV arrays or path strings.

### 3. Build the VLM Instruction Dataset

Files:

```text
ml/vlm/scripts/build_instruction_dataset.py
ml/vlm/scripts/validate_instruction_dataset.py
ml/vlm/tests/test_instruction_dataset.py
```

Outputs:

```text
results/vlm/instruction_smoke/train.jsonl
results/vlm/instruction_smoke/valid.jsonl
results/vlm/instruction_smoke/summary.json
```

JSONL row shape:

```json
{
  "sample_id": "...",
  "split": "train",
  "images": ["Train/.../sample.png"],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "Train/.../sample.png"},
        {"type": "text", "text": "...Output JSON only..."}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"label_id\":3,\"diagnosis\":\"corona_discharge\",...}"
    }
  ]
}
```

### 4. SmolVLM2 Inference Smoke

File:

```text
ml/vlm/src/service_adapter.py
```

Validation command:

```powershell
python -m pytest service/backend/tests/test_model_runtime.py
```

Success criteria:

- Adapter registry resolves the VLM manifest and checkpoint.
- Image + text input is processed successfully.
- Raw output is returned or fallback errors are recorded clearly.
- JSON parses successfully or parse errors are recorded clearly.
- Runtime metadata records checkpoint/model status.

### 5. QLoRA SFT Smoke

Files:

```text
ml/vlm/train.py
docs/VLM_TRAINING_GUIDE.md
```

Validation command:

```powershell
python ml/vlm/train.py `
  --model-profile smolvlm2_2b_qlora `
  --manifest data/manifest.csv `
  --sample-size 20 `
  --epochs 1 `
  --max-steps 20 `
  --eval-steps 5 `
  --gpu-memory-fraction 0.9 `
  --output-dir artifacts/models/vlm/smolvlm2_2b_qlora
```

Success criteria:

- 20-step smoke training completes.
- Adapter checkpoint is saved.
- TensorBoard event file is saved.
- Training summary is saved.
- If OOM occurs, 8GB fallback settings are recorded.

### 6. JSON Evaluation

File:

```text
ml/vlm/scripts/evaluate_outputs.py
```

Evaluation command:

```powershell
python ml/vlm/scripts/evaluate_outputs.py `
  --predictions results/vlm/predictions_smoke.jsonl `
  --output results/vlm/eval_smoke.json
```

Required metrics:

```text
json_parse_success_rate
schema_validity_rate
label_accuracy
macro_f1
confusion_matrix
forbidden_field_hit_count
hallucinated_field_count
```

## Experiment Stages

### Stage 0: Zero-Shot / Few-Shot

```text
sample: 20
training: none
purpose: validate processor, prompt, and JSON-output feasibility
```

### Stage 1: LoRA Smoke

```text
sample: 10~100
model: Qwen3-VL-2B
training: LoRA SFT
purpose: confirm the training loop works on 8GB VRAM
```

### Stage 2: QLoRA Small

```text
sample: 500~2,000
model: Qwen3-VL-2B
training: 4-bit QLoRA SFT
purpose: check JSON parse rate and label accuracy
```

### Stage 3: Qwen2.5-VL-3B Comparison

```text
sample: 500~2,000
model: Qwen2.5-VL-3B
training: 4-bit QLoRA SFT
condition: only after 2B runs stably
```

### Stage 4: Main

```text
sample: 5,000~10,000
model: choose the more stable model
condition: after confirming JSON parse success and VRAM stability
```

### Stage 5: Strategy B

```text
input: PRPD PNG + waveform/spectrogram PNG
condition: only after Strategy A succeeds
```

## Strategy A/B

### Strategy A: Implement First

```text
1 PRPD image
+ safe metadata
+ time-series summary
-> VLM
-> JSON diagnosis
```

This is the default strategy.

### Strategy B: Lower Priority

```text
1 PRPD image
+ 1 time-series waveform/spectrogram image
+ safe metadata
+ time-series summary
-> VLM
-> JSON diagnosis
```

Cautions:

- Two images increase visual-token cost.
- OOM risk is higher on 8GB VRAM.
- Try only after Strategy A works.

## Final Implementation Checklist

- [ ] Create the `ml/vlm/` directory.
- [ ] Implement prompt/schema code.
- [ ] Implement forbidden-field leakage tests.
- [ ] Implement time-series context export.
- [ ] Implement instruction dataset builder.
- [ ] Implement instruction dataset validator.
- [ ] Implement Qwen3-VL-2B inference smoke.
- [ ] Implement QLoRA SFT smoke.
- [ ] Implement JSON evaluation.
- [ ] Update model priority in `docs/VLM_STRATEGY.md`.
- [ ] Write `docs/VLM_DEVELOPMENT_RUNBOOK.md`.

## Success Criteria

- VLM input consists of PRPD image + safe metadata + time-series summary.
- Raw CSV and label-leakage fields are absent from prompts.
- Qwen3-VL-2B inference smoke is possible.
- LoRA/QLoRA smoke training works on 8GB VRAM, or clear fallback settings are recorded.
- Output JSON can be parsed and evaluated automatically.
- Strategy A is validated first, and Strategy B remains lower priority.
