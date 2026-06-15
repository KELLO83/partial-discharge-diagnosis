# partial-discharge-diagnosis

Portfolio project for an end-to-end partial-discharge composite diagnosis system.

The project connects five layers:

- time-series model development for partial-discharge CSV signals
- vision and VLM development for PRPD image, metadata, and model-context reasoning
- FastAPI diagnosis service with guardrails, trace storage, review actions, and report export
- React factory-manager dashboard for inspection, review queue, case timeline, and model status
- documentation and LLM wiki notes for the model-to-agent development path

## Quick Start

Use the repository root `.env` for runtime settings. For a deterministic demo,
set `MODEL_ADAPTER_MODE=mock`. For local checkpoint inference, set
`MODEL_ADAPTER_MODE=checkpoint` or `auto` and fill the `MODEL_*_MANIFEST` /
`MODEL_*_CHECKPOINT` paths shown below.

Backend:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r service/backend/requirements.txt
$env:MODEL_ADAPTER_MODE = "mock"
python -m uvicorn service.backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd service/frontend
bun install
bun run dev
```

Open the Vite URL printed by the frontend, usually `http://127.0.0.1:5173`.

Optional RAG setup:

```powershell
# Edit DATABASE_URL in .env first.
python service/backend/scripts/rag_init_db.py
python service/backend/scripts/rag_ingest_sources.py
```

Useful checks:

```powershell
python -m pytest service/backend/tests/test_model_runtime.py
cd service/frontend
bun run typecheck
```

Current limitation: this is a solo portfolio project and uses the available local dataset only. External MES, ERP, PLC, CMMS, and large-scale factory integrations are represented as architecture notes or demo workflow surfaces, not as deployed production integrations.

Current runtime state: the service supports both mock adapters and checkpoint-backed adapters. The local baseline checkpoints are now connected through the manifest registry:

| Task | Current baseline | Artifact |
|---|---|---|
| Time-series | `inception_time_small` | `artifacts/models/time_series/inception_time_small/20260615_194838/best.pt` |
| Vision | `efficientnet_b0` | `artifacts/models/vision/efficientnet_b0/20260615_201805/best.pt` |
| VLM | `smolvlm2_2b_qlora` | `artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/best.pt` |

Runtime configuration is centralized in the repository root `.env`. Database URLs, RAG settings, OpenRouter keys, model adapter mode, model artifact paths, and the frontend API base should be changed there instead of setting per-service environment files.

Model serving is wired through a manifest-based adapter registry. Use `MODEL_ADAPTER_MODE=mock` for deterministic demo tests, or `MODEL_ADAPTER_MODE=checkpoint` / `auto` for real local artifacts. Direct `.env` overrides are supported:

```dotenv
MODEL_TIME_SERIES_MANIFEST=artifacts/models/time_series/inception_time_small/20260615_194838/model_manifest.json
MODEL_TIME_SERIES_CHECKPOINT=artifacts/models/time_series/inception_time_small/20260615_194838/best.pt
MODEL_VISION_MANIFEST=artifacts/models/vision/model_manifest.json
MODEL_VISION_CHECKPOINT=artifacts/models/vision/efficientnet_b0/20260615_201805/best.pt
MODEL_VLM_MANIFEST=artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/model_manifest.json
MODEL_VLM_CHECKPOINT=artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/best.pt
```

The frontend does not read checkpoint files directly. It calls the backend model-status and diagnosis APIs; the backend resolves manifests, checkpoints, adapters, and runtime mode.

RAG is production-shaped but still lightweight. The service uses PostgreSQL + pgvector under the `partial_discharge_diagnosis` database with `rag.documents`, `rag.chunks`, and `rag.query_logs`. Text retrieval is configured for `dragonkue/multilingual-e5-small-ko-v2` with 384-dimensional vectors, while deterministic fallback embeddings keep the local demo usable before the embedding dependency is installed. The customer dashboard does not expose RAG administration; RAG runs behind the diagnosis workflow as evidence retrieval for the LLM/VLM report.

LLM RAG can call OpenRouter through the OpenAI-compatible chat completions API. Keep `LLM_RAG_PROVIDER=auto`; when `OPENROUTER_API_KEY` is present in root `.env`, the service uses OpenRouter for the final RAG-grounded diagnosis report, and otherwise falls back to the local development VLM adapter.

Default RAG sources are rulebook markdown and dataset case summaries from `data/manifest.csv`. SOP markdown can be enabled later through `RAG_SOURCE_TYPES`; maintenance manuals are intentionally excluded.

Current model-training docs:

- project folder/code role map: `docs/PROJECT_STRUCTURE.md`
- VLM training and re-training: `docs/VLM_TRAINING_GUIDE.md`
- VLM strategy: `docs/VLM_STRATEGY.md`
- VLM implementation plan: `docs/VLM_IMPLEMENTATION_PLAN.md`
- VLM runbook: `docs/VLM_DEVELOPMENT_RUNBOOK.md`
- time-series model families: `docs/TIMESERIES_MODELS.md`
- local project memory: `docs/llm_wiki/`

Local virtual environments, raw AI Hub data, external model repositories, checkpoints, uploads, and experiment outputs are intentionally excluded from Git.
