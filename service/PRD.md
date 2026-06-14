# PRD: Partial-Discharge Diagnosis Service

## 1. Purpose

The service layer provides an end-to-end diagnosis skeleton before final model training is complete.

It must already run like the target product:

```text
PRPD PNG
+ time-series CSV
+ equipment/environment metadata
-> input validation
-> time-series evidence adapter
-> lightweight vision evidence adapter
-> similar dataset case retrieval adapter
-> RAG knowledge retrieval adapter
-> VLM report adapter
-> reviewer guardrails
-> case trace, report, manager dashboard
```

Current adapters are deterministic mocks. After model work is complete, only the adapter implementations should need replacement.

## 2. Non-Goals

- Do not train models in the service layer.
- Do not run model experiments from the dashboard.
- Do not put raw CSV rows, image binaries, label names, sample IDs, or file paths into prompts or traces.
- Do not make the VLM the only diagnostic model.

## 3. Runtime Components

### FastAPI Backend

Location:

```text
service/backend/app/
```

Responsibilities:

- receive multipart diagnosis requests
- validate PNG and CSV shape
- parse safe metadata
- call deterministic model/retrieval adapters
- apply confidence, probability, disagreement, and RAG guardrails
- store trace, history, review actions, comments, and report detail

### React Manager Dashboard

Location:

```text
service/frontend/src/
```

Responsibilities:

- upload PRPD PNG and time-series CSV
- edit equipment metadata
- show final verdict
- show metadata, time-series, vision, similar case, RAG, and VLM evidence cards
- show trace log, history, review queue, model runtime status, similar cases, and report references
- provide a dedicated similar-case search page for field operators to filter historical PRPD cases by label, equipment, sensor, and insulation type
- provide a RAG administration panel below the trace menu for index health, evidence search, source counts, recent query logs, and reindexing
- render all operator-facing copy in Korean because the target user is a plant/process manager or PRPD monitoring operator

## 4. Adapter Contracts

All model and retrieval adapters must expose both native outputs and a common service evidence contract. The common contract keeps the dashboard, report generator, reviewer, and future checkpoint-backed models independent from model-specific feature names.

Common model evidence:

```text
source
model_name
model_version
label_id
label_name
confidence
uncertainty
ood_score
top_factors[]
explanation
```

Each `top_factors[]` item contains:

```text
name, value, weight, explanation
```

Model-specific raw features may still exist, but operator-facing evidence must come through this standard contract.

### Time-Series Adapter

Input:

```text
CSV artifact path + sha256
```

Output:

```text
label_id, label_name, confidence, probabilities, summary features
+ standard evidence factors for signal amplitude, pulse rate, and spectral energy
```

Current mock:

```text
mock_patchtst@pre_model_mock
```

Service deployment contract:

```text
artifacts/models/time_series/model_manifest.json
-> entrypoint: ml.timeseries.src.service_adapter:load_adapter
-> backend method: predict_csv(TimeSeriesToolInput) -> dict
-> normalized output: TimeSeriesResult
```

### Vision Adapter

Input:

```text
PRPD PNG artifact path + sha256
```

Output:

```text
label_id, label_name, confidence, probabilities, visual evidence, OOD hint
+ standard evidence factors for phase localization, noise band, and PRPD OOD score
```

Current mock:

```text
mock_prpd_small_cnn@pre_model_mock
```

Service deployment contract:

```text
artifacts/models/vision/model_manifest.json
-> entrypoint: ml.vision.src.service_adapter:load_adapter
-> backend method: predict_image(VisionToolInput) -> dict
-> normalized output: VisionResult
```

### RAG Retrieval Adapter

Input:

```text
safe metadata + time-series result + vision result
+ similar dataset case result
```

Output:

```text
query, retriever name/version, compact evidence documents, reference cases
```

Current mock:

```text
pgvector_rulebook_case_rag@dragonkue_multilingual_e5_small_ko_v2
```

Implementation:

```text
PostgreSQL database: partial_discharge_diagnosis
schema: rag
embedding model: dragonkue/multilingual-e5-small-ko-v2
embedding dimension: 384
default sources: rulebook markdown + dataset case summaries
optional source: SOP markdown, enabled only through RAG_SOURCE_TYPES
excluded source: maintenance manuals
```

The first production-shaped RAG implementation uses PostgreSQL + pgvector. LangGraph remains a later internal RAG orchestration option; it should not replace the current diagnosis workflow until the retrieval boundary is stable.

RAG responsibilities:

- retrieve rulebook evidence for the current candidate label and sensor context
- retrieve SOP evidence for review, remeasurement, and uncertainty handling
- retrieve text summaries of historical dataset cases
- rerank vector hits with candidate label, sensor, equipment type, and insulation metadata so operationally relevant evidence is promoted
- provide evidence to the VLM reporter, fusion evidence contract, reviewer guardrails, and the dashboard evidence card
- expose administration APIs for status, document listing, query logs, manual search, and reindexing
- not directly vote for the final diagnosis label

### Similar Case Retrieval Adapter

Input:

```text
safe metadata + time-series result + vision result
```

Output:

```text
Top 5 dataset cases with label, PRPD image URL, metadata, similarity score, and retrieval reason
```

Current mock:

```text
mock_dataset_case_retriever@pre_embedding_mock
```

This is the practical field-evidence path: a new upload can be compared against existing `data/` cases before trained embedding models are available.

The same dataset case repository also powers the dashboard's manual similar-case search:

```text
label / equipment / sensor / insulation / free-text query
-> matching historical PRPD references
```

### VLM Adapter

Input:

```text
PRPD PNG artifact
+ safe metadata
+ time-series evidence
+ vision evidence
+ RAG evidence
```

Output:

```text
diagnosis label, risk level, confidence, reason, recommended action
+ standard evidence factors summarizing model agreement, retrieved rulebook evidence, and top similar case
```

Current mock:

```text
mock_qwen3_vl_2b@pre_model_mock
```

Service deployment contract:

```text
artifacts/models/vlm/model_manifest.json
-> entrypoint: ml.vlm.src.service_adapter:load_adapter
-> backend method: generate_report(VlmToolInput) -> dict
-> normalized output: VlmResult
```

Adapter mode:

```text
MODEL_ADAPTER_MODE=mock       # use deterministic service mocks
MODEL_ADAPTER_MODE=checkpoint # require all model artifacts to be ready
MODEL_ADAPTER_MODE=auto       # use ready checkpoint adapters and mock the rest
```

## 5. Workflow

```text
input_router
-> metadata_context
-> time_series_tool
-> vision_tool
-> similar_case_tool
-> rag_tool
-> vlm_tool
-> fusion_engine
-> diagnosis_reviewer
-> report_agent
```

Route behavior:

- `hybrid`: PNG + valid CSV + valid metadata
- `vlm_only`: PNG + valid metadata
- `timeseries_only`: valid CSV only
- `insufficient_input`: no valid diagnostic input

The RAG tool runs on every non-rejected route. The VLM tool runs only when image and metadata are available.

Fusion engine behavior:

```text
collect standard evidence from time-series, vision, VLM, RAG, and similar cases
-> compute agreement level
-> summarize contributing sources
-> expose a fusion rationale to trace/report/dashboard
```

The first implementation is deterministic rule-based late fusion. A trained meta-classifier can replace it later without changing the frontend evidence contract.

## 6. Guardrails

Input guardrails:

- CSV shape must match the expected service shape.
- PRPD image must be PNG-like.
- metadata must reject label leakage and unexpected fields.

Tool guardrails:

- time-series and vision label IDs must be valid.
- probability outputs must contain classes `0` through `4` and sum near `1`.
- confidence must meet the release threshold.
- Similar case retrieval should return dataset references when the local manifest is available.
- Fusion should state whether available models agree, partially agree, or conflict.
- RAG query must be non-empty.
- RAG must return at least one sufficiently relevant document.
- RAG database unavailability must fall back to deterministic local evidence so the service remains demonstrable before production DB provisioning.
- VLM reason and recommendation must be non-empty.

Reviewer behavior:

```text
if invalid input:
    rejected
elif low confidence:
    needs_review
elif time-series / vision / VLM labels disagree:
    needs_review
elif RAG evidence is missing or weak:
    needs_review
else:
    completed
```

## 7. API Surface

Core endpoints:

- `GET /health`
- `GET /model-status`
- `POST /diagnose`
- `GET /diagnose/{diagnosis_id}/trace`
- `GET /diagnoses`
- `GET /diagnoses/{diagnosis_id}`
- `GET /review-queue`
- `POST /diagnoses/{diagnosis_id}/actions`
- `POST /diagnoses/{diagnosis_id}/comments`
- `GET /diagnoses/{diagnosis_id}/report`
- `GET /dataset/cases`
- `GET /dataset/cases/search`
- `GET /dataset/cases/{sample_id}`
- `GET /dataset/cases/{sample_id}/image`

Scenario replay endpoints:

- `GET /demo/scenarios`
- `POST /demo/seed`
- `POST /demo/scenarios/{scenario_id}/activate`

## 8. Done Criteria

- A user can upload PNG + CSV + metadata from the dashboard.
- Backend returns a diagnosis response and trace.
- Trace includes metadata, time-series, vision, similar case, RAG, VLM, reviewer, and report events for hybrid input.
- Trace includes standardized model evidence and fusion rationale.
- Manager dashboard renders Korean operator-facing copy, intake upload, evidence board, current similar cases, history, detail, review queue, and report download.
- RAG can initialize a PostgreSQL + pgvector schema and ingest rulebook, SOP, and dataset case summary chunks.
- The service can run without trained model checkpoints and without a populated RAG database by using deterministic fallback evidence.
- Backend and frontend tests pass.
- The service can keep running while future trained adapters are developed under `ml/timeseries`, `ml/vision`, and `ml/vlm`.
