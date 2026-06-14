# PRD: Partial-Discharge Composite Diagnosis System

## 1. Purpose

This repository builds a service-grade partial-discharge diagnosis system from the AI-Hub industrial electrical fire prevention dataset.

The root project goal is not one monolithic model. The system is split into three ML tracks and one service layer:

```text
CSV time-series signal
+ PRPD image
+ safe equipment/environment metadata
+ rulebook/SOP/case retrieval evidence
-> time-series model
-> lightweight vision model
-> standard evidence contract
-> rule-based late fusion
-> VLM report generator
-> guardrailed diagnosis service
```

Each ML track owns its own detailed PRD and agent rules because input format, model design, evaluation, and runtime constraints differ.

## 2. Dataset

Local working data lives under `Train/`.

Current `data/manifest.csv` summary:

- Total samples: `30,010`
- Balanced labels: `6,002` samples for each label `0` through `4`
- Each sample links:
  - `timeseries_path`: partial-discharge CSV time-series
  - `image_path`: PRPD PNG image
  - `json_path`: label and metadata JSON
- CSV shape: `(20, 7680)`

Label mapping:

| ID | Class |
| --- | --- |
| 0 | normal |
| 1 | noise |
| 2 | surface_discharge |
| 3 | corona_discharge |
| 4 | void_discharge |

`data/manifest.csv` is the source of truth for future model training and service-side dataset case retrieval. Some manifest path values may still start with `Train/` as an original extraction prefix; service code resolves those paths against the current `data/` folder.

## 3. ML Tracks

### 3.1 Time-Series Track

Location:

```text
ml/timeseries/
```

Responsibility:

```text
CSV tensor (20, 7680)
-> time-series / feature model
-> 5-class prediction, probabilities, confidence, evidence features
```

Detailed docs:

- `ml/timeseries/PRD.md`
- `ml/timeseries/AGENTS.md`

### 3.2 Vision Track

Location:

```text
ml/vision/
```

Responsibility:

```text
PRPD PNG or normalized PRPD tensor
-> lightweight vision model
-> 5-class prediction, confidence, visual evidence, OOD hint
```

This track starts with lightweight models, not a large VLM and not a full vision-tower LoRA experiment.

Detailed docs:

- `ml/vision/PRD.md`
- `ml/vision/AGENTS.md`

### 3.3 VLM Track

Location:

```text
ml/vlm/
```

Responsibility:

```text
PRPD image
+ safe metadata
+ time-series evidence
+ vision evidence
-> structured JSON diagnosis report
```

The VLM is a report generator and consistency checker. It must not be treated as the only diagnostic model.

Detailed docs:

- `ml/vlm/PRD.md`
- `ml/vlm/AGENTS.md`

## 4. Service Layer

Location:

```text
service/
```

Responsibility:

```text
input validation
-> model tool calls
-> rulebook/SOP retrieval
-> similar dataset case retrieval
-> disagreement / confidence / OOD guardrails
-> diagnosis record
-> trace and dashboard
```

The service may use mock adapters until trained checkpoints are connected.

## 5. Architecture Rule

Do not collapse all diagnosis logic into the VLM.

Preferred flow:

```text
time-series model evidence
+ lightweight vision model evidence
+ metadata
+ retrieved rulebook/SOP evidence
-> VLM explanation/report
-> reviewer policy
```

The VLM explains and formats evidence. The time-series and vision tracks produce the primary diagnostic signals.

## 6. Leakage Rules

Never use these as model features or VLM prompt text:

- file paths
- file names
- sample IDs
- label names
- `PD_type`
- defect details
- full raw CSV rows in prompts
- any metadata column proven to encode the label directly

`label_id` is allowed in supervised targets and evaluation records only.

## 7. Success Criteria

Minimum success:

- `ml/timeseries` has a working classifier baseline.
- `ml/vision` has a lightweight PRPD vision baseline.
- `ml/vlm` can generate valid structured JSON from model evidence.
- The service can show a traceable diagnosis workflow with agreement, disagreement, and review cases.

Strong success:

- Time-series and vision models are evaluated on the same split.
- VLM reports cite model evidence and respect low-confidence/OOD flags.
- All model adapters expose common `standard_evidence` with top factors and operator-facing explanation text.
- Fusion summarizes agreement/conflict across time-series, vision, VLM, RAG, and similar cases before the reviewer step.
- RAG evidence grounds reports in rulebook, SOP, or past-case snippets.
- Similar Case Retrieval shows Top 5 dataset references with PRPD image, label, metadata, similarity score, and retrieval reason, and also supports a manual operator search page.
- Disagreement cases route to review instead of being finalized.
- The frontend is Korean-first and reads as a plant/process manager console, not a developer dashboard.
- Documentation stays separated by track so model-specific design does not crowd the root PRD.

## 8. Document Map

Root:

- `PRD.md`: project-level architecture and ownership
- `AGENT.md`: repository-level coding and execution rules

Track-level:

- `ml/timeseries/PRD.md`
- `ml/timeseries/AGENTS.md`
- `ml/vision/PRD.md`
- `ml/vision/AGENTS.md`
- `ml/vlm/PRD.md`
- `ml/vlm/AGENTS.md`

Reference docs:

- `docs/DATASET_EXPLAIN.md`
- `docs/TIMESERIES_MODELS.md`
- `docs/VLM_STRATEGY.md`
- `docs/VLM_IMPLEMENTATION_PLAN.md`
- `service/PRD.md`
