# RAG Chat Modules

`service.backend.app.rag.chat` keeps the public RAG chat API while separating the implementation by role.

| File | Role |
| --- | --- |
| `service.py` | Orchestrates guard, retrieval, OpenRouter completion, and response mapping. |
| `models.py` | Defines request input and injectable dependencies for tests/runtime. |
| `guard.py` | Decides whether a question should skip RAG retrieval. |
| `prompts.py` | Builds system/user prompts and RAG context text. |
| `parser.py` | Extracts and validates the JSON payload returned by the LLM. |
| `constants.py` | Stores chat limits, domain terms, and local fallback text. |
| `__init__.py` | Re-exports the stable public API used by FastAPI and tests. |
