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
- Project structure map: `docs/PROJECT_STRUCTURE.md`
- Dataset explanation: `docs/DATASET_EXPLAIN.md`
- Time-series model plan: `docs/TIMESERIES_MODELS.md`
- Model implementation source policy: `docs/MODEL_IMPLEMENTATION_SOURCES.md`
- VLM strategy: `docs/VLM_STRATEGY.md`
- VLM implementation plan: `docs/VLM_IMPLEMENTATION_PLAN.md`
- VLM development runbook: `docs/VLM_DEVELOPMENT_RUNBOOK.md`
- VLM training guide: `docs/VLM_TRAINING_GUIDE.md`

## Stable Implementation Boundaries

| Boundary | Code |
|---|---|
| Model artifact registry | `service/backend/app/models/model_artifacts.py` |
| Runtime adapter selection | `service/backend/app/models/model_runtime.py` |
| Checkpoint adapters | `service/backend/app/models/checkpoint_adapters.py` |
| Local agent runtime | `service/backend/app/application/agent_runtime.py` |
| Workflow boundary | `service/backend/app/application/workflow.py` |
| Upload artifacts | `service/backend/app/infrastructure/artifacts.py` |
| Trace storage | `service/backend/app/infrastructure/store.py` |
| Shared TS features | `ml/timeseries/src/features/timeseries_summary.py` |
| VLM service adapter | `ml/vlm/src/service_adapter.py` |
| VLM training entrypoint | `ml/vlm/train.py` |

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
| Qwen-VL family | Future VLM comparison family | `source_cards/models/qwen_vl.md` |
| Time-series model families | Candidate TS model map | `source_cards/models/time_series_models.md` |

## Experiment Notes

Experiment notes now include the first checkpoint-backed baseline runs. Use:

```text
experiment_notes/time_series/
experiment_notes/vlm/
experiment_notes/composite_eval/
experiment_notes/vision/
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
