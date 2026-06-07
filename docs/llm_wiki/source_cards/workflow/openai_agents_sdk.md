# OpenAI Agents SDK

## Source

- Title: OpenAI Agents SDK
- URL: `https://openai.github.io/openai-agents-python/`

## Relevance

The final service can use the SDK for agent orchestration, tool execution,
guardrails, handoffs, and tracing. Current implementation uses a deterministic
local runtime first so the service remains testable without API credentials.

## Adopt / Defer / Avoid

- Adopt: Agent/Runner orchestration after deterministic service contracts are
  stable.
- Adopt: SDK tracing for production observability when API-backed orchestration
  is enabled.
- Defer: Handoffs until workflow branching becomes conversational or complex.
- Avoid: Letting an LLM invent labels outside validated tool outputs.

## Current Project Boundary

`service/backend/app/openai_agents_adapter.py` currently checks availability.
`service/backend/app/agent_runtime.py` is the default local runtime.
