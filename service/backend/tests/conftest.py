from __future__ import annotations

import os


os.environ["LLM_RAG_PROVIDER"] = "mock"
os.environ.pop("OPENROUTER_API_KEY", None)
