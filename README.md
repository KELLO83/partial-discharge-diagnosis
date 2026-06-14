# partial-discharge-diagnosis

Portfolio project for an end-to-end partial-discharge composite diagnosis system.

The project connects five layers:

- time-series model development for partial-discharge CSV signals
- VLM planning for PRPD image, metadata, and time-series summary reasoning
- FastAPI diagnosis service with guardrails, trace storage, review actions, and report export
- React factory-manager dashboard for inspection, review queue, case timeline, and model status
- documentation and LLM wiki notes for the model-to-agent development path

Current limitation: this is a solo portfolio project and uses the available local dataset only. External MES, ERP, PLC, CMMS, and large-scale factory integrations are represented as architecture notes or demo workflow surfaces, not as deployed production integrations.

Current service adapters are intentionally mock-backed until trained model checkpoints are connected. After model development, replace the mock time-series and VLM adapters with checkpoint-backed adapters; the admin dashboard, trace API, review workflow, and report export are already structured around that handoff.

Model serving is wired through a manifest-based adapter registry. Keep `MODEL_ADAPTER_MODE=mock` while checkpoints are absent. When ML artifacts are ready, copy the templates from `service/backend/model_artifact_templates/<task>/model_manifest.json` into `artifacts/models/<task>/model_manifest.json`, place the referenced checkpoint/preprocessor files next to the manifest, implement the matching `ml/<task>/src/service_adapter.py` entrypoint, and set `MODEL_ADAPTER_MODE=checkpoint` or `auto`.

RAG is production-shaped but still lightweight. The service uses PostgreSQL + pgvector under the `partial_discharge_diagnosis` database with `rag.documents`, `rag.chunks`, and `rag.query_logs`. Text retrieval is configured for `dragonkue/multilingual-e5-small-ko-v2` with 384-dimensional vectors, while deterministic fallback embeddings keep the local demo usable before the embedding dependency is installed. The customer dashboard does not expose RAG administration; RAG runs behind the diagnosis workflow as evidence retrieval for the LLM/VLM report.

LLM RAG can call OpenRouter through the OpenAI-compatible chat completions API. Keep `LLM_RAG_PROVIDER=auto`; when `OPENROUTER_API_KEY` is present, the service uses OpenRouter for the final RAG-grounded diagnosis report, and otherwise falls back to the local development VLM adapter.

Local RAG setup:

```powershell
$env:DATABASE_URL="postgresql://postgres:<password>@localhost:5432/partial_discharge_diagnosis"
python service/backend/scripts/rag_init_db.py
python service/backend/scripts/rag_ingest_sources.py
```

Default RAG sources are rulebook markdown and dataset case summaries from `data/manifest.csv`. SOP markdown can be enabled later through `RAG_SOURCE_TYPES`; maintenance manuals are intentionally excluded.

Local virtual environments, raw AI Hub data, external model repositories, checkpoints, uploads, and experiment outputs are intentionally excluded from Git.
