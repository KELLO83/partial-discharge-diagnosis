# Composite Diagnosis Architecture

## System Flow

```text
React
-> FastAPI /diagnose
-> upload artifact storage
-> local deterministic agent runtime
-> time-series tool
-> VLM tool
-> reviewer guardrails
-> report response
-> trace endpoint
```

## Why Local Runtime Exists

OpenAI Agents SDK is the future orchestration target. The local runtime exists
so that service tests, model adapter development, and offline evaluation work
without API keys or SDK installation.

This keeps model development independent from agent orchestration dependencies.

## Tool Contracts

The time-series tool receives:

```text
csv_path
csv_sha256
```

The VLM tool receives:

```text
image_path
image_sha256
safe_metadata
timeseries_result
```

The tool contracts intentionally avoid raw CSV content, image bytes, label
fields from metadata, sample IDs, and file-name-derived labels.

## Reviewer Contract

The reviewer may only choose between validated tool outputs. It must not invent
new labels.

Main branches:

- `completed`: tool outputs are valid and consistent.
- `needs_review`: low confidence, disagreement, invalid probabilities, or VLM
  output concerns.
- `rejected`: invalid input or no usable tool output.

## Trace Contract

Trace events record:

- input route
- tool summaries
- model names and versions
- labels and confidence
- artifact checksums
- reviewer decision
- report decision

Trace events must not store raw CSV rows, PRPD image bytes, checkpoints, or full
VLM prompts containing sensitive fields.
