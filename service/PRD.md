# PRD: Partial-Discharge Diagnosis Service Agent Workflow

## 1. Purpose

This document defines the future `React + FastAPI + OpenAI Agents SDK` partial-discharge diagnosis workflow. This stage is service planning before implementation. It targets inference-service orchestration, not model training.

The service goal is: when a user provides a PRPD image, time-series CSV, and equipment/environment metadata, FastAPI runs an Agents SDK workflow, calls existing time-series and VLM inference models as tools, and returns the final diagnosis report.

```text
React
-> FastAPI
-> Agents SDK workflow
-> Time-series inference tool
-> VLM inference tool
-> Guardrail / Reviewer
-> Final diagnosis report
```

## 2. Scope

### Included

- React input-screen design direction
- FastAPI diagnosis API design direction
- Agents SDK diagnosis workflow
- Time-series model inference tool
- VLM inference tool
- Input/output guardrails
- Trace/audit log design
- Human-review branching conditions

### Excluded

- Time-series model training
- VLM model training
- QLoRA training-code changes
- Real industrial-site deployment
- User authentication/authorization
- Payment or operations-admin features

## 3. User Scenarios

### Scenario A: Normal Diagnosis Request

The user enters the following in the React screen:

- PRPD PNG image
- Partial-discharge time-series CSV
- Equipment name
- Rated voltage/current
- Insulator information
- Sensor type
- Temperature/humidity
- Clearance distance

The service returns:

```json
{
  "diagnosis_id": "diag_20260604_000001",
  "status": "completed",
  "final_label_id": 3,
  "diagnosis": "corona_discharge",
  "risk_level": "caution",
  "confidence": 0.87,
  "reason": "The time-series model and VLM diagnosis both support a corona-discharge interpretation.",
  "recommended_action": "Inspect high-voltage connection points and electric-field concentration areas, then monitor the trend.",
  "requires_human_review": false
}
```

### Scenario B: Input Error

If the CSV shape is not `(20, 7680)` or the image is not PNG, the workflow does not start.

```json
{
  "status": "rejected",
  "error_code": "INVALID_INPUT",
  "message": "timeseries_csv must have shape (20, 7680)."
}
```

### Scenario C: Low Confidence

If the time-series model and VLM disagree or confidence is low, the service does not finalize the diagnosis and returns review status.

```json
{
  "status": "needs_review",
  "requires_human_review": true,
  "reason": "The time-series model and VLM predicted different labels."
}
```

## 4. Overall Architecture

```text
frontend/
  React app
  - upload form
  - metadata form
  - diagnosis result view
  - trace view

service/
  FastAPI backend
  - upload endpoint
  - diagnosis endpoint
  - trace endpoint
  - Agents SDK workflow

ml/
  time-series inference code

vlm/
  VLM inference code
  prompt builder
  output evaluator
```

## 5. API Design

### POST `/diagnose`

Runs the diagnosis workflow.

Request:

```text
multipart/form-data
- prpd_image: PNG
- timeseries_csv: CSV
- metadata: JSON string
```

Metadata JSON:

```json
{
  "equipment_name": "ACSR-OC",
  "equipment_rated_voltage": "22900V",
  "equipment_rated_current": "268A",
  "insulator_type": "solid",
  "insulator_name": "XLPE",
  "sensor_type": "HFCT",
  "temperature": 19,
  "humidity": 66,
  "clearance_distance": "1000mm"
}
```

Response:

```json
{
  "diagnosis_id": "diag_...",
  "status": "completed | needs_review | rejected",
  "final_label_id": 0,
  "diagnosis": "normal",
  "risk_level": "low",
  "confidence": 0.91,
  "reason": "...",
  "recommended_action": "...",
  "requires_human_review": false,
  "trace_id": "trace_..."
}
```

### GET `/diagnose/{diagnosis_id}`

Returns a stored diagnosis result.

### GET `/diagnose/{diagnosis_id}/trace`

Returns the Agent workflow execution trace and tool-call results.

### GET `/health`

Returns service status, model loading status, and GPU availability.

## 6. Agents SDK Design

The OpenAI Agents SDK is a diagnosis-process manager, not a trainer. Apply its core concepts, including Agent, tool, handoff, guardrail, and tracing, to the service workflow.

### 6.1 Orchestrator Agent

Responsibilities:

- Start the full diagnosis workflow.
- Check input-validation results.
- Call the time-series tool.
- Call the VLM tool.
- Call the Reviewer Agent.
- Call the final Report Agent.

Instructions:

```text
You are the workflow manager for partial-discharge diagnosis.
Do not invent labels directly. Build the final judgment only from tool results.
If the time-series model and VLM results conflict, do not finalize a diagnosis. Branch to needs_review.
```

### 6.2 Data Intake Agent

Responsibilities:

- Validate uploaded file metadata.
- Interpret CSV shape validation results.
- Interpret image-format validation results.
- Check required input metadata fields.
- Block potential label leakage.

This Agent does not run model inference. If input is invalid, it stops the workflow.

### 6.3 Time-Series Inference Tool

This is a deterministic tool called by the Agent.

Input:

```json
{
  "timeseries_csv_path": "uploads/diag_x/signal.csv"
}
```

Output:

```json
{
  "model_name": "patchtst",
  "label_id": 3,
  "label_name": "corona_discharge",
  "confidence": 0.87,
  "probabilities": {
    "0": 0.02,
    "1": 0.04,
    "2": 0.06,
    "3": 0.87,
    "4": 0.01
  },
  "features": {
    "rms": 30.37,
    "std": 4.96,
    "abs_p99": 39.0,
    "pulse_rate": 0.0069,
    "spectral_energy": 13982100.0
  }
}
```

Cautions:

- Do not put the full raw CSV in an Agent or VLM prompt.
- Provide only inference results and summary features to the Agent.

### 6.4 VLM Inference Tool

This is a deterministic tool called by the Agent.

Input:

```json
{
  "prpd_image_path": "uploads/diag_x/prpd.png",
  "safe_metadata": {
    "equipment_name": "ACSR-OC",
    "equipment_rated_voltage": "22900V",
    "sensor_type": "HFCT",
    "temperature": 19,
    "humidity": 66
  },
  "timeseries_summary": {
    "ts_pred_class": 3,
    "ts_confidence": 0.87,
    "rms": 30.37,
    "std": 4.96,
    "abs_p99": 39.0,
    "pulse_rate": 0.0069,
    "spectral_energy": 13982100.0
  }
}
```

Output:

```json
{
  "label_id": 3,
  "diagnosis": "corona_discharge",
  "risk_level": "caution",
  "reason": "The PRPD image and time-series summary are consistent with a corona-discharge pattern.",
  "recommended_action": "Inspect high-voltage connection points and electric-field concentration areas."
}
```

Candidate models:

- Local first: `Qwen/Qwen3-VL-2B-Instruct`
- Local fallback: `Qwen/Qwen2.5-VL-3B-Instruct`
- Higher VRAM/cloud comparison: `LGAI-EXAONE/EXAONE-4.5-33B-AWQ`

### 6.5 Diagnosis Reviewer Agent

Responsibilities:

- Compare time-series model and VLM results.
- Validate VLM JSON schema.
- Detect label mismatch.
- Check confidence threshold.
- Block exaggerated recommended actions.
- Decide whether human review is required.

Branching rules:

```text
if input_validation_failed:
    status = rejected
elif ts_confidence < 0.60:
    status = needs_review
elif ts_label_id != vlm_label_id:
    status = needs_review
elif vlm_json_schema_invalid:
    status = needs_review
else:
    status = completed
```

### 6.6 Report Agent

Responsibilities:

- Generate the final user response.
- Normalize diagnosis JSON.
- Generate a short explanation for field engineers.
- Summarize why human review is required when applicable.

The Report Agent cannot create a new diagnosis label. It only formats results approved by the Reviewer Agent.

## 7. Guardrail Design

### Input Guardrail

Checks before workflow start:

- File extension validation
- Image MIME validation
- CSV shape validation
- Required metadata field validation
- Detection of user-supplied labels

### Tool Guardrail

Checks before and after each tool call.

Time-Series Tool:

- Confirm the input CSV path stays inside the upload directory.
- Confirm output `label_id` is in range `0` through `4`.
- Confirm probabilities sum close to 1.

VLM Tool:

- Confirm forbidden fields are absent from the prompt.
- Confirm output JSON is parseable.
- Confirm all required keys exist.

### Output Guardrail

Checks before final response:

- Confirm `status` is an allowed enum.
- Confirm `final_label_id` maps to `diagnosis`.
- If `requires_human_review=true`, avoid definitive recommended actions.

## 8. Trace / Audit Log

Use Agents SDK tracing to record:

- `diagnosis_id`
- `trace_id`
- Input file validation results
- Time-Series Tool input/output summary
- VLM Tool input/output summary
- Reviewer Agent judgment
- Final response
- Human-review branch reason

Sensitive-data policy:

- Do not store the full raw CSV in traces.
- Do not store PRPD image binaries in traces.
- Store only paths, checksums, shapes, and summary features.

## 9. Data Security and Leakage Prevention

Values forbidden in prompts:

- `label_id`
- `label_name`
- `PD_type`
- `sample_id`
- file name
- file path
- `defect_details`
- `defect_nums`
- `max_discharge_value`

Service input metadata must not accept target labels. Target labels exist only in training/evaluation data and are not included in service inference requests.

## 10. React Screen Design

### Diagnose Page

Inputs:

- PRPD image upload
- Time-series CSV upload
- Equipment/environment metadata form
- Run diagnosis button

Outputs:

- Final diagnosis label
- Risk level
- Confidence
- Evidence/reason
- Recommended action
- Human-review status

### Trace Page

Shows:

- Input validation status
- Time-series model result
- VLM result
- Reviewer judgment
- Final response generation time

## 11. FastAPI Service Structure

Recommended modules:

```text
service/
  PRD.md
  backend/
    app/
      main.py
      api/
        diagnose.py
        health.py
      schemas/
        request.py
        response.py
      agents/
        workflow.py
        prompts.py
        guardrails.py
      tools/
        timeseries.py
        vlm.py
      storage/
        uploads.py
        traces.py
  frontend/
    src/
      pages/
        DiagnosePage.tsx
        TracePage.tsx
```

## 12. Phased Implementation Plan

### Phase 1: API Skeleton

- FastAPI `/health`
- FastAPI `/diagnose`
- request/response schemas
- file-upload storage

### Phase 2: Deterministic Inference Tools

- Time-Series Inference Tool
- VLM Inference Tool
- tool-level mock/stub support

### Phase 3: Agents SDK Workflow

- Orchestrator Agent
- Data Intake Agent
- Diagnosis Reviewer Agent
- Report Agent
- Decide between handoff and manager-as-tools structures

Recommended approach:

```text
initial implementation: manager-as-tools
extended implementation: handoff
```

Reason:

- The initial service has a fixed diagnosis workflow, so a manager that calls tools in order is more predictable.
- Introduce handoffs when the service becomes conversational or has more complex workflow branching.

### Phase 4: React UI

- upload form
- metadata form
- diagnosis result view
- trace view

### Phase 5: QA and Deployment Preparation

- valid-sample diagnosis end-to-end
- invalid CSV rejection
- low-confidence review branch
- time-series/VLM disagreement review branch
- trace-storage validation

## 13. Definition of Done

- React can upload PRPD image, CSV, and metadata.
- FastAPI `/diagnose` receives the request and runs the workflow.
- The Time-Series Tool runs CSV inference and returns summary features.
- The VLM Tool receives PRPD image plus safe context and returns diagnosis JSON.
- The Reviewer Agent detects disagreement, low confidence, and invalid JSON.
- Final response status is one of `completed`, `needs_review`, or `rejected`.
- `/diagnose/{id}/trace` can return workflow trace data.

## 14. References

- OpenAI Agents SDK GitHub: `https://github.com/openai/openai-agents-python`
- Agents SDK agents guide: `https://openai.github.io/openai-agents-python/agents/`
- Agents SDK handoffs guide: `https://openai.github.io/openai-agents-python/handoffs/`
- Agents SDK guardrails reference: `https://openai.github.io/openai-agents-python/ref/guardrail/`
- Agents SDK tracing guide: `https://openai.github.io/openai-agents-python/tracing/`
- Existing project PRD: `docs/PRD.md`
- VLM runbook: `docs/VLM_DEVELOPMENT_RUNBOOK.md`
