# Project Structure

This document explains where each major code path belongs. Use it as the first
map before changing runtime, model training, or frontend behavior.

## Top-Level Folders

| Folder | Role |
|---|---|
| `service/backend/` | FastAPI diagnosis service, runtime adapters, workflow orchestration, persistence, RAG, tests |
| `service/frontend/` | React dashboard for diagnosis intake, case detail, report, review queue, model status |
| `ml/timeseries/` | Time-series model training, feature extraction, dataset utilities, exported checkpoints |
| `ml/vision/` | PRPD image classifier training and vision checkpoint artifacts |
| `ml/vlm/` | VLM instruction dataset creation, QLoRA/SFT training, VLM service adapter |
| `ml/training/` | Shared training artifact helpers such as timestamped model directories |
| `artifacts/models/` | Local model manifests, checkpoints, processors, TensorBoard event files |
| `data/` | Local manifest and dataset references; raw external data stays out of Git |
| `docs/` | PRDs, model strategy, runbooks, training guides, project memory |
| `docs/llm_wiki/` | Agent-readable project state, source cards, experiment notes, development gates |

## Backend Service

| Path | Role |
|---|---|
| `service/backend/app/main.py` | FastAPI app wiring and HTTP endpoints |
| `service/backend/app/application/workflow.py` | Diagnosis workflow boundary |
| `service/backend/app/application/agent_runtime.py` | Deterministic local orchestration runtime |
| `service/backend/app/application/contracts.py` | Tool input/output contracts for time-series, vision, VLM, RAG |
| `service/backend/app/models/model_artifacts.py` | Manifest loading, `.env` artifact overrides, readiness records |
| `service/backend/app/models/model_runtime.py` | Selects mock/checkpoint adapters and exposes runtime status info |
| `service/backend/app/models/checkpoint_adapters.py` | Normalizes checkpoint backend outputs into service schemas |
| `service/backend/app/models/mock_adapters.py` | Deterministic mock adapters for tests and local demos |
| `service/backend/app/domain/` | Domain policy, fusion evidence, reviewer logic |
| `service/backend/app/infrastructure/` | File artifacts, trace storage, database/RAG infrastructure |
| `service/backend/tests/` | Backend unit and API tests |

## VLM Track

| Path | Role |
|---|---|
| `ml/vlm/train.py` | Main CLI: builds instruction JSONL and runs dry-run or QLoRA/SFT training |
| `ml/vlm/src/model_profiles.py` | Supported VLM training profiles and guardrail defaults |
| `ml/vlm/src/prompts.py` | Prompt construction and forbidden-field control |
| `ml/vlm/src/schema.py` | VLM labels, dataset rows, target JSON schema |
| `ml/vlm/src/service_adapter.py` | Runtime adapter used by backend checkpoint serving |
| `ml/vlm/scripts/build_instruction_dataset.py` | Converts manifest/context CSVs into instruction JSONL |
| `ml/vlm/scripts/export_ts_context.py` | Creates compact time-series context CSV |
| `ml/vlm/scripts/export_vision_context.py` | Creates compact vision context CSV |
| `ml/vlm/scripts/train_sft.py` | Low-level SFT/QLoRA training implementation |
| `ml/vlm/scripts/evaluate_outputs.py` | JSON and label evaluation for generated VLM outputs |
| `ml/vlm/tests/` | VLM prompt, dataset, training-guardrail, and evaluation tests |

## Runtime Model Flow

```text
root .env
-> ModelAdapterSettings.from_env()
-> ModelArtifactRegistry
-> ModelArtifactRecord per task
-> build_service_model_runtime()
-> mock or checkpoint adapter
-> diagnosis workflow
-> frontend model-status/report views
```

The frontend never opens checkpoints directly. It only calls backend APIs. The
backend is responsible for resolving manifests, checkpoints, preprocessors, and
adapter mode.

## Training Artifact Flow

```text
training CLI
-> artifacts/models/<task>/<model>/<run_id>/
-> best.pt
-> model_manifest.json
-> TensorBoard event files
-> root .env MODEL_*_MANIFEST / MODEL_*_CHECKPOINT
-> backend runtime
```

Keep raw datasets, virtual environments, downloaded foundation models, and local
checkpoints out of Git unless an artifact is intentionally sanitized for docs.
