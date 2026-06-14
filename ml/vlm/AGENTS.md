# AGENTS: VLM Track

## Scope

Work in this folder when changing instruction datasets, prompt templates, VLM inference, SFT/QLoRA training, JSON validation, or VLM report evaluation.

Primary paths:

```text
ml/vlm/src/
ml/vlm/train.py
ml/vlm/scripts/
ml/vlm/tests/
ml/vlm/configs/
```

## Role

The VLM is a report generator and consistency checker.

It should consume evidence from:

```text
ml/timeseries
ml/vision
safe metadata
PRPD image
```

It should not be the only diagnostic model.

## Rules

- Do not put full raw CSV values in prompts.
- Do not put path strings, filenames, sample IDs, label names, `PD_type`, or defect details in prompts.
- Keep output JSON strict and schema-valid.
- Include uncertainty and `needs_review` when evidence is low-confidence, OOD, or contradictory.
- Freeze the vision encoder for first experiments.
- Use QLoRA/SFT mainly for output format, terminology, and report style.

## Imports

Use:

```python
from ml.vlm.src.prompts import build_prompt_text
from ml.timeseries.src.features.timeseries_summary import summarize_signal
```

Do not use old imports:

```python
from vlm...
from ml.src...
```

## Commands

Dataset build smoke:

```powershell
python ml/vlm/scripts/export_ts_context.py --manifest data/manifest.csv --sample-size 20 --output .omo/evidence/vlm-ts-context.csv
python ml/vlm/train.py --model-profile smolvlm2_2b_qlora --manifest data/manifest.csv --sample-size 20 --ts-context .omo/evidence/vlm-ts-context.csv --dry-run
python ml/vlm/scripts/validate_instruction_dataset.py --input artifacts/models/vlm/instruction_dataset.jsonl --output .omo/evidence/vlm-validate.json
```

Tests:

```powershell
pytest ml/vlm/tests
```

## Output

Write outputs under:

```text
results/vlm/
.omo/evidence/
```

Do not store raw image bytes, raw CSV rows, or full sensitive prompts in service traces.
