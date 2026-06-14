# VLM Implementation Plan

## Goal

The VLM stage is not PRPD image-only classification. The goal is to build a model that receives the following inputs and generates structured diagnosis JSON:

```text
PRPD image
+ safe equipment/environment metadata
+ time-series model prediction
+ time-series summary features
-> lightweight pretrained VLM
-> diagnosis JSON
```

The target GPU is an RTX 4060 Laptop with 8GB VRAM. Full fine-tuning of large VLMs is excluded. Fine-tune a lightweight pretrained VLM with LoRA/QLoRA.

## Model Selection

### First Choice: Qwen/Qwen3-VL-2B-Instruct

This is the first model to validate on the 8GB GPU.

Reasons:

- The 2B scale is the most realistic for 8GB VRAM.
- It is an image-text-to-text VLM, matching the PRPD image + text metadata input design.
- Qwen-family models are generally strong at instruction following and JSON-style output.
- It is suitable for LoRA/QLoRA SFT experiments.

### Stable Alternative: Qwen/Qwen2.5-VL-3B-Instruct

Use as a comparison candidate if Qwen3-VL-2B is unstable locally or quality is insufficient.

Cautions:

- The 3B scale is tighter on 8GB VRAM.
- Use 4-bit QLoRA, batch size 1, gradient accumulation, and gradient checkpointing.

### Lower-Priority Risk Candidate: Qwen/Qwen3-VL-4B-Instruct

Review only after 2B/3B smoke tests pass.

Cautions:

- OOM risk is high on 8GB VRAM.
- It is not the first implementation target.
- 4-bit QLoRA, low LoRA rank, and image-resolution limits are required.

### Fallback Candidates

- `HuggingFaceTB/SmolVLM2-2.2B-Instruct`: use if Qwen-family models are blocked by memory or installation issues.
- `google/paligemma2-3b-mix-224`: possible image-text transfer candidate, but Qwen-family models remain the priority for JSON diagnosis output.
- `llava-hf/llava-onevision-qwen2-0.5b-si-hf`: pipeline sanity-check candidate only.

## Input Data Design

Do not feed raw CSV directly into the VLM. Use one image and one text context.

### Strategy A: Default Input

```text
image:
  PRPD PNG from manifest.image_path

text:
  equipment information
  environment information
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

### Time-Series Model Information for Text Input

After time-series experiments are complete, export the following information into a separate CSV and join it in the VLM dataset builder:

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

If time-series model results are unavailable, the smoke dataset should explicitly state that the time-series model result is unavailable and use feature-only context.

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
model_id: Qwen/Qwen3-VL-2B-Instruct
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
flash_attention: false
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

### 4. Qwen3-VL-2B Inference Smoke

File:

```text
ml/vlm/scripts/run_inference.py
```

Validation command:

```powershell
python ml/vlm/scripts/run_inference.py `
  --dataset results/vlm/instruction_smoke/valid.jsonl `
  --index 0 `
  --model-id Qwen/Qwen3-VL-2B-Instruct `
  --load-in-4bit `
  --output results/vlm/inference_smoke.json
```

Success criteria:

- Model loads successfully.
- Image + text input is processed successfully.
- Raw output is saved.
- JSON parses successfully or parse errors are recorded clearly.
- CUDA peak memory is recorded.

### 5. QLoRA SFT Smoke

Files:

```text
ml/vlm/scripts/train_sft.py
ml/vlm/configs/qwen3_vl_2b_smoke.yaml
```

Validation command:

```powershell
python ml/vlm/scripts/train_sft.py `
  --config ml/vlm/configs/qwen3_vl_2b_smoke.yaml `
  --max-steps 10
```

Success criteria:

- 10-step smoke training completes.
- Adapter checkpoint is saved.
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
