# VLM Strategy

This document defines the VLM development direction for the partial-discharge project.

The VLM is not an image-only model that predicts a discharge type from a PRPD image alone. The goal is to combine multimodal information and generate a diagnosis that a field engineer can understand.

```text
PRPD image
+ JSON metadata
+ optional vision model prediction/context
+ time-series model prediction
+ time-series summary features
-> natural-language or JSON diagnosis report
```

## Default Direction

Image-only classification is not the core VLM scope.

Excluded direction:

```text
PRPD image -> ResNet/EfficientNet -> 5-class classification
```

Target direction:

```text
PRPD image + equipment/environment metadata + time-series summary
-> Small VLM
-> structured diagnosis JSON or natural-language diagnosis
```

The key VLM task is multimodal diagnosis reporting, not image classification.

## Recommended Models

### Current Implemented Baseline: SmolVLM2-2.2B-Instruct

Use `HuggingFaceTB/SmolVLM2-2.2B-Instruct` as the current local VLM baseline.

Reasons:

- It is small enough for RTX 4060 Laptop 8GB QLoRA smoke training.
- It supports the project input structure: PRPD image plus text context.
- It completed the current local training smoke path and produced a checkpoint.
- It works with PyTorch SDPA, so FlashAttention is not required.

Current service artifact:

```text
artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/best.pt
```

### Future Comparison Candidates

- `Qwen/Qwen2.5-VL-3B-Instruct`: good next comparison once the SmolVLM2 baseline
  has a larger validation run.
- `Qwen/Qwen3-VL-2B-Instruct`: keep as a candidate if local Transformers support
  and Windows dependencies are stable.
- `google/paligemma2-3b-mix-224`: possible transfer candidate, but less aligned
  with strict JSON diagnosis reports than instruction-tuned VLMs.

### Not Current Targets

- `Qwen/Qwen3-VL-4B-Instruct`: high OOM risk on the current 8GB GPU target.
- `LGAI-EXAONE/EXAONE-4.0-1.2B-AWQ`: text-only LLM, not a VLM.
- Large EXAONE VLM variants: too heavy for the current local baseline workflow.

## Input Data Design

VLM input combines an image and text.

Image input:

```text
PRPD PNG image
```

Text input:

```text
equipment information
- equipment name
- insulator type
- rated voltage
- rated current
- sensor type

environment information
- temperature
- humidity
- clearance distance

time-series analysis information
- time-series model predicted class ID
- confidence
- class probability
- RMS
- std
- abs_p99
- pulse_rate
- spectral energy

vision analysis information, when available
- vision model predicted class ID
- confidence
- class probability
```

Do not place full raw CSV data in VLM prompts. Compress raw time-series signals through a time-series model or feature extractor and provide text features. Do not include `label_id`, `label_name`, file paths, class-bearing file names, defect-detail fields, or `max_discharge_value` in user prompts.

## Output Format

Initial training should prioritize JSON output over free-form natural language because JSON is easier to evaluate and post-process.

Recommended output example:

```json
{
  "label_id": 1,
  "diagnosis": "noise",
  "risk_level": "low",
  "reason": "The PRPD pattern and time-series features are closer to a noise-like signal than to actual partial discharge.",
  "recommended_action": "Check sensor contact and nearby electromagnetic interference."
}
```

Evaluation items:

- `label_id` accuracy
- `diagnosis` label-name match rate
- JSON parse success rate
- metadata-use checks
- time-series-information-use checks
- hallucination checks
- diagnosis-text quality

## Training Method

Start with QLoRA-based SFT.

Recommended initial settings:

```text
base_model: HuggingFaceTB/SmolVLM2-2.2B-Instruct
quantization: 4bit NF4
training: SFT
vision_encoder: freeze
projector: freeze or partial LoRA
LLM: LoRA
batch_size: 1
gradient_accumulation_steps: 8~16
gradient_checkpointing: enabled
attention_backend: PyTorch SDPA
gpu_memory_fraction: 0.9
```

Do not train the full vision encoder at the beginning. The PRPD domain may be unfamiliar to the pretrained VLM, but pretrained vision encoders can still extract generic visual features such as points, lines, density, distribution, and symmetry.

Recommended order:

```text
1. Freeze the vision encoder.
2. Apply LoRA to language-model layers.
3. Train JSON target generation from image + metadata + time-series summaries.
4. If performance is insufficient, review LoRA on the projector or selected vision-encoder layers.
```

## Data-Scale Strategy

The current `Train/` working dataset contains 30,010 samples, which is sufficient for VLM practice and early LoRA experiments.

Recommended stages:

```text
1. VLM smoke: 100~500 samples
2. First LoRA: 2,000~5,000 samples
3. Main experiment: 10,000~30,000 samples
4. Final extension: part or all of the original 300k samples
```

Do not start with all 300k samples. First validate data format, training stability, and JSON-output quality on the 30k working dataset.

## Connection to Vision and Time-Series Models

VLM development does not strictly require the time-series and vision classifiers
to be trained first. A VLM can learn from PRPD image plus safe metadata alone.
For the composite diagnosis system, however, the preferred runtime input also
includes compact classifier context.

Active classifier artifacts:

```text
time-series: inception_time_small
vision: efficientnet_b0
```

Context fields:

```text
best model name
predicted label
confidence
class probabilities
statistical features
optional embedding
```

Include available information in the VLM instruction prompt as text context.

Example prompt:

```text
Equipment information:
- equipment_name: ACSR-OC
- insulator: solid / XLPE
- rated_voltage: 22900V
- sensor_type: HFCT

Environment information:
- temperature: 19 C
- humidity: 66%

Time-series model analysis:
- predicted_class_id: 1
- confidence: 0.82
- RMS: 0.221
- STD: 0.140
- abs_p99: 1.300
- pulse_rate: 0.004

Vision model analysis:
- predicted_class_id: 1
- confidence: 0.77

Using the attached PRPD image and the information above, diagnose the current partial-discharge state as JSON.
```

## Project Storyline

Final portfolio flow:

```text
1. Build CSV time-series classification models.
2. Build a lightweight PRPD image classifier baseline.
3. Extract compact predictions and summaries from the best classifiers.
4. Convert PRPD image + safe metadata + classifier context into a VLM instruction dataset.
5. Fine-tune SmolVLM2-2.2B-Instruct with QLoRA.
6. Generate and evaluate JSON diagnosis reports.
```

Core message:

```text
This is not a simple image classifier. It is an explainable industrial partial-discharge diagnosis system that connects time-series sensor analysis and equipment metadata to a VLM.
```
