from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.backend.app.rag.ingest import initialize_rag_database


def main() -> None:
    initialize_rag_database()
    print("RAG database schema initialized.")


if __name__ == "__main__":
    main()
