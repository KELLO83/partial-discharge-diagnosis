# LLM Wiki Index

## Scope

This wiki tracks the composite partial-discharge diagnosis project:

```text
CSV time-series model
+ PRPD image
+ safe equipment/environment metadata
+ VLM JSON diagnosis
+ deterministic reviewer/guardrails
+ Agent service trace
-> completed / needs_review / rejected diagnosis
```

Chronological wiki changes are recorded in `LOG.md`.

## Required Local Context

- Main project PRD: `PRD.md`
- Detailed PRD: `docs/PRD.md`
- Pre-model infrastructure PRD: `docs/PRE_MODEL_DEVELOPMENT_PRD.md`
- Service PRD: `service/PRD.md`
- Dataset explanation: `docs/DATASET_EXPLAIN.md`
- Time-series model plan: `docs/TIMESERIES_MODELS.md`
- Model implementation source policy: `docs/MODEL_IMPLEMENTATION_SOURCES.md`
- VLM strategy: `docs/VLM_STRATEGY.md`
- VLM implementation plan: `docs/VLM_IMPLEMENTATION_PLAN.md`
- VLM development runbook: `docs/VLM_DEVELOPMENT_RUNBOOK.md`

## Stable Implementation Boundaries

| Boundary | Code |
|---|---|
| Time-series adapter | `service/backend/app/tool_contracts.py` |
| VLM adapter | `service/backend/app/tool_contracts.py` |
| Local agent runtime | `service/backend/app/agent_runtime.py` |
| Reviewer guardrails | `service/backend/app/guardrails.py` |
| Upload artifacts | `service/backend/app/artifacts.py` |
| Trace storage | `service/backend/app/store.py` |
| Offline evaluator | `service/backend/app/offline.py` |
| Shared TS features | `ml/timeseries/src/features/timeseries_summary.py` |

## Concept Notes

| Concept | Note |
|---|---|
| Current development findings | `concepts/current_development_findings.md` |
| Composite diagnosis architecture | `concepts/composite_diagnosis_architecture.md` |
| Model development gates | `concepts/model_development_gates.md` |

## Source Cards

| Source / tool | Role | Card |
|---|---|---|
| OpenAI Agents SDK | Future SDK orchestration target | `source_cards/workflow/openai_agents_sdk.md` |
| Qwen-VL family | Primary VLM candidate family | `source_cards/models/qwen_vl.md` |
| Time-series model families | Candidate TS model map | `source_cards/models/time_series_models.md` |

## Experiment Notes

Experiment notes are empty until real model runs begin. Use:

```text
experiment_notes/time_series/
experiment_notes/vlm/
experiment_notes/composite_eval/
experiment_notes/vision_optional/
```

Each note must state sample size, split, command, checkpoint, metrics, and
limitations.

## Reproducibility Snapshots

Sanitized evidence copied from ignored `results/` belongs under:

```text
docs/llm_wiki/experiment_notes/artifacts/
```

Do not store raw CSVs, PRPD image binaries, checkpoints, or private model
weights in the wiki.
