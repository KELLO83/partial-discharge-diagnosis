# AGENTS: Vision Track

## Scope

Work in this folder when developing lightweight PRPD vision models, PRPD image datasets, image transforms, visual evidence exports, or vision evaluation.

Primary paths:

```text
ml/vision/src/
ml/vision/train.py
ml/vision/scripts/
ml/vision/tests/
```

## First Implementation Bias

Start small:

```text
PRPD PNG
-> 224x224 tensor
-> small CNN
-> 5-class classifier
```

Do not start with large ViT, VLM vision LoRA, or multi-image VLM training.

## Rules

- Use `data/manifest.csv` as the source of truth.
- Load images from `image_path`, but never expose path strings as model text features.
- Keep labels from `label_id`.
- Prefer deterministic train/valid split behavior shared with `ml/timeseries`.
- Keep transforms simple and reproducible.
- Record image size, normalization, augmentation, and model name in every run output.
- Treat plot color, font, grid, and axis style as possible shortcuts.

## Allowed Models

Initial:

```text
small CNN
ResNet18
MobileNetV3-Small
EfficientNet-B0
```

Avoid initially:

```text
large ViT
Qwen-VL vision tower LoRA
image-only production claims
```

## Imports

Use:

```python
from ml.vision.src...
```

Do not put vision code under `ml.timeseries` or `ml.vlm`.

## Validation

Expected future commands:

```powershell
pytest ml/vision/tests
python ml/vision/train.py --sample-size 100 --epochs 1 --dry-run
python ml/vision/train.py --sample-size 500 --epochs 5
```

When scripts are added, they must support smoke subsets before full-dataset runs.

## Output

Write outputs under:

```text
results/vision/
```

Vision evidence exported to VLM/service should use stable numeric fields and short evidence summaries, not raw image bytes.
