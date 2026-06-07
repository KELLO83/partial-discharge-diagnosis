# Qwen-VL Family

## Source

- Title: Qwen-VL model family
- Role: Primary local VLM candidate family for PRPD image + text context.

## Relevance

The project plans to use Qwen-family VLMs for structured JSON diagnosis from:

```text
PRPD image + safe metadata + time-series summary
```

## Candidate Order

1. `Qwen/Qwen3-VL-2B-Instruct`
2. `Qwen/Qwen2.5-VL-3B-Instruct`
3. Larger variants only after local smoke tests are stable.

## Adopt / Defer / Avoid

- Adopt: QLoRA SFT, JSON-first output, frozen vision encoder initially.
- Defer: full vision encoder tuning.
- Avoid: raw CSV in prompts and label-leaking metadata fields.
