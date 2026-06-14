# AGENT: Repository-Level Coding Rules

## 1. Purpose

This document defines repository-level rules for agents working on the partial-discharge diagnosis system.

Detailed ML rules are track-specific:

```text
ml/timeseries/AGENTS.md
ml/vision/AGENTS.md
ml/vlm/AGENTS.md
```

Before changing a track, read that track's `PRD.md` and `AGENTS.md`.

## 2. Project Structure

```text
ml/
  timeseries/   CSV time-series and feature models
  vision/       lightweight PRPD vision models
  vlm/          multimodal report generator

service/
  FastAPI, frontend dashboard, diagnosis workflow, traces

docs/
  dataset, model, and implementation references

Train/
  local dataset source files
```

## 3. Root Architecture Rule

Do not collapse the full diagnosis system into one VLM.

Preferred architecture:

```text
time-series evidence
+ lightweight vision evidence
+ safe metadata
+ retrieved rulebook/SOP evidence
-> VLM report generator
-> guardrailed service workflow
```

The VLM explains and formats evidence. It does not replace the time-series or vision models.

## 4. Data Rules

Use `Train/manifest.csv` as the source of truth for local experiments.

Never use these as features or VLM prompt text:

- file paths
- file names
- sample IDs
- label names
- `PD_type`
- defect details
- full raw CSV rows in prompts
- metadata proven to encode labels directly

`label_id` is allowed in supervised targets and evaluation records only.

## 5. Directory Boundaries

Use the correct track:

- Time-series code: `ml/timeseries/`
- Lightweight PRPD vision code: `ml/vision/`
- VLM prompt/training/report code: `ml/vlm/`
- Service workflow and UI code: `service/`
- RAG/retrieval adapter contracts and mock service evidence: `service/backend/app/`

Do not put vision-model code under `ml/timeseries`.
Do not put VLM SFT/report code under `ml/vision`.
Do not put CSV model training code under `ml/vlm`.

## 6. Imports

Use current package paths:

```python
from ml.timeseries.src...
from ml.vision.src...
from ml.vlm.src...
```

Do not use old package paths:

```python
from ml.src...
from ml.scripts...
from vlm...
```

## 7. Runtime and Tests

Default Python environment:

```text
.venv
```

Common tests:

```powershell
pytest ml/timeseries/tests
pytest ml/vision/tests
pytest ml/vlm/tests
```

Optional external model tests may skip when official packages or cloned repositories are unavailable. They must not silently substitute unrelated model implementations.

## 8. Outputs

Write generated outputs under:

```text
results/
.omo/evidence/
```

Never overwrite raw source files under `Train/`.

Do not store raw CSV rows, image binaries, private checkpoints, or full sensitive prompts in service traces.
