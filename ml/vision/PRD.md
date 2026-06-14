# PRD: Lightweight Vision Model Track

## 1. Purpose

This track develops a lightweight PRPD vision model that is separate from both CSV time-series models and VLM report generation.

Initial goal:

```text
PRPD PNG or normalized PRPD tensor
-> lightweight vision classifier
-> 5-class prediction, confidence, and visual evidence
-> evidence context for ml/vlm and service guardrails
```

This is intentionally not a large VLM, not a full vision-tower LoRA experiment, and not the final explanation layer.

## 2. Input

Primary input:

```text
data/manifest.csv image_path
PRPD PNG image
```

Preferred normalized input:

```text
PRPD image
-> crop/resize
-> remove plot-style dependence where practical
-> tensor [C, H, W]
```

Future numeric input:

```text
CSV/raw signal
-> phase-amplitude histogram
-> PRPD tensor [phase_bins, amplitude_bins]
```

Start with image input because it is available now, but design the code so tensor-based PRPD features can replace or augment PNG later.

## 3. Output

Required output:

```text
vision_model_name
vision_pred_label_id
vision_confidence
vision_prob_0..vision_prob_4
```

Recommended evidence output:

```text
phase_localization_score
phase_uniformity_score
band_like_noise_score
symmetry_hint
ood_score
visual_evidence_summary
```

Downstream export:

```text
results/vision/evidence_context.csv
```

Service adapter contract:

```text
ml/vision/src/service_adapter.py
load_adapter(context) -> backend
backend.predict_image(VisionToolInput) -> dict
```

The returned dict must contain `label_id`, `confidence`, `probabilities`, and `evidence`. The service normalizes this into `VisionResult`.

## 4. First Model Strategy

Use lightweight models first:

```text
small custom CNN
ResNet18
MobileNetV3-Small
EfficientNet-B0
ConvNeXt-Tiny only if memory/runtime is acceptable
```

Recommended first implementation:

```text
PRPD PNG 224x224
-> SmallPrpdCnn
-> classifier head
-> calibration/evidence export
```

Reason:

- Fast smoke tests
- Low VRAM/CPU cost
- Easy debugging
- Clear baseline before heavier vision encoders

## 5. Training Order

1. Build image dataset from `data/manifest.csv`.
2. Add leakage checks for path/name/label text.
3. Train small CNN smoke on a class-balanced subset.
4. Evaluate on the same split as `ml/timeseries`.
5. Export vision evidence context.
6. Compare against time-series predictions.
7. Only then try ResNet18 or MobileNetV3-Small.

## 6. Evaluation

Required metrics:

```text
accuracy
macro_f1
per_class_recall
confusion_matrix
normal_vs_noise_confusion
noise_vs_PD_confusion
```

Robustness checks:

- random split
- equipment-aware split if available
- sensor-aware split if available
- image style sensitivity
- metadata removed

## 7. Non-Goals

- Do not build the VLM here.
- Do not fine-tune Qwen-VL vision towers here.
- Do not treat screenshot style as trusted physics.
- Do not use label-bearing filenames or paths as text features.
- Do not optimize only for validation accuracy without checking domain/style shortcuts.

## 8. Integration

`ml/vision` produces evidence for:

```text
ml/vlm
service/backend
offline composite evaluation
```

The VLM may use the vision result as context, but the VLM should not overwrite high-confidence vision/time-series disagreement without review.

## 9. Current CLI

Dry-run:

```powershell
python ml/vision/train.py --sample-size 20 --epochs 1 --dry-run
```

Training:

```powershell
python ml/vision/train.py --sample-size 500 --epochs 5
```

Default outputs:

```text
artifacts/models/vision/checkpoint.pt
artifacts/models/vision/model_manifest.json
artifacts/models/vision/evidence_context.csv
artifacts/models/vision/train_summary.json
```
